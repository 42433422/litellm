"""The router's side of ``dynamic_queue``: the ``router_settings.dynamic_queue`` block, the queue time each request
owns across its attempts, per-attempt admission, and the public errors a refused wait raises.

Enabled, every async chat or text completion attempt on a deployment the router owns waits for its capacity's turn,
and its outcome is reported back. Only a failure whose provider answered with one of ``trigger_status_codes`` congests
a capacity. A refused wait ends the request with no retry, fallback or cooldown.
"""

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable, Coroutine, Generator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator
from typing_extensions import Self, assert_never

from litellm.exceptions import ServiceUnavailableError, Timeout
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.total_deadline import TotalDeadline, current_total_deadline
from litellm.router_utils.dynamic_queue import (
    OPERATION_TIMED_OUT,
    Admitted,
    Attempt,
    CoordinationUnavailable,
    DynamicQueue,
    LocalQueueBackend,
    MatchingFailure,
    MonotonicTimer,
    OtherFailure,
    Outcome,
    PassThrough,
    QueueBackend,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Rejection,
    Timer,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_redis import RedisQueueBackend, ScriptHost

CAPACITY_ID_FIELD: Final = "dynamic_queue_capacity_id"
REDIS_NOT_CONFIGURED: Final = "redis_not_configured"
QUEUE_FULL_CODE: Final = "queue_full"
QUEUE_COORDINATION_UNAVAILABLE_CODE: Final = "queue_coordination_unavailable"

_HttpErrorStatus: TypeAlias = Annotated[StrictInt, Field(ge=400, le=599)]


class DynamicQueueSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    trigger_status_codes: tuple[_HttpErrorStatus, ...] = (429, 529)
    queue_timeout: float = Field(default=30.0, gt=0)
    max_waiting: int = Field(default=1_000, gt=0)
    coordination: Literal["redis", "local"] = "redis"
    recovery_successes: int = Field(default=2, ge=1)
    initial_backoff: float = Field(default=1.0, gt=0)
    max_backoff: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _backoff_grows_to_its_cap(self) -> Self:
        if self.max_backoff < self.initial_backoff:
            raise ValueError("dynamic_queue.max_backoff must be at least dynamic_queue.initial_backoff")
        return self

    def queue_settings(self) -> QueueSettings:
        return QueueSettings(
            required_successes=self.recovery_successes,
            initial_backoff_ms=math.ceil(self.initial_backoff * 1000),
            max_backoff_ms=math.ceil(self.max_backoff * 1000),
            max_waiting=self.max_waiting,
        )


_SETTINGS: Final[TypeAdapter[DynamicQueueSettings | None]] = TypeAdapter(DynamicQueueSettings | None)


def parse_dynamic_queue(value: object, default_priority: int | None) -> DynamicQueueSettings | None:
    settings: Final = _SETTINGS.validate_python(value)
    if settings is None or not settings.enabled:
        return None
    if default_priority is None:
        raise ValueError("router_settings.dynamic_queue.enabled requires router_settings.default_priority")
    return settings


def refuse_local_queue_across_workers(value: object, num_workers: str) -> None:
    settings: Final = _SETTINGS.validate_python(value)
    if settings is not None and settings.enabled and settings.coordination == "local" and num_workers != "1":
        raise ValueError(
            f"router_settings.dynamic_queue.coordination 'local' serves one process, not {num_workers} workers; "
            "use 'redis'"
        )


class QueueRejectedError(Exception):
    """Marks the public errors below so the router ends the request without a retry, fallback or cooldown."""


class DynamicQueueTimeoutError(Timeout, QueueRejectedError):
    pass


class DynamicQueueFullError(ServiceUnavailableError, QueueRejectedError):
    def __init__(self, model: str) -> None:
        super().__init__(message="the deployment's dynamic queue is full", llm_provider="", model=model)
        self.code = QUEUE_FULL_CODE


class DynamicQueueCoordinationError(ServiceUnavailableError, QueueRejectedError):
    def __init__(self, model: str) -> None:
        super().__init__(message="the dynamic queue's coordination is unavailable", llm_provider="", model=model)
        self.code = QUEUE_COORDINATION_UNAVAILABLE_CODE


def rejection_error(rejection: Rejection, model: str) -> QueueRejectedError:
    match rejection:
        case QueueTimeout():
            return DynamicQueueTimeoutError(
                message="the request ran out of dynamic queue time", model=model, llm_provider=""
            )
        case QueueFull():
            return DynamicQueueFullError(model)
        case CoordinationUnavailable():
            return DynamicQueueCoordinationError(model)
        case _:
            assert_never(rejection)


_T: Final = TypeVar("_T")


async def _within(timer: Timer, seconds: float, call: Coroutine[object, object, _T]) -> _T | None:
    work: Final = asyncio.create_task(call)
    guard: Final = asyncio.create_task(timer.sleep(seconds))
    try:
        _ = await asyncio.wait((work, guard), return_when=asyncio.FIRST_COMPLETED)
        return work.result() if work.done() else None
    finally:
        for task in (work, guard):
            _ = task.cancel()
        _ = await asyncio.gather(work, guard, return_exceptions=True)


@dataclass(frozen=True, slots=True)
class Admission:
    """An admitted attempt, held until its provider call answers or its stream ends."""

    attempt: Attempt
    held: contextlib.AsyncExitStack
    router: "DynamicQueueRouter"

    async def finish(self, outcome: Outcome | None) -> None | CoordinationUnavailable:
        """What reporting the outcome came to: a CoordinationUnavailable means the queue never learned it."""
        reported: Final = None if outcome is None else await self.attempt.report(outcome)
        await self.held.aclose()
        return reported

    async def went_idle(self, reads: Callable[[], "StreamReads | None"]) -> bool:
        """Whether a stream sat a whole idle bound with no read in progress and none finished while it still held its
        turn; ``reads`` is None once the stream settled its turn itself."""
        idle_s: Final = self.router.queue_settings.lease_ms * 3 / 1000
        while await self._read_during(idle_s, reads):
            pass
        return reads() is not None

    async def _read_during(self, idle_s: float, reads: Callable[[], "StreamReads | None"]) -> bool:
        before: Final = reads()
        if before is None:
            return False
        await self.router.timer.sleep(idle_s)
        after: Final = reads()
        return after is not None and (after.in_progress or after.done != before.done)


@dataclass(frozen=True, slots=True)
class StreamReads:
    done: int
    in_progress: bool


class _Backend(QueueBackend, Protocol):
    async def busy(self, capacity: str) -> bool | CoordinationUnavailable: ...


class RequestQueue:
    """The queue state one request owns across its attempts and its stream's recovery. Only time spent waiting is
    taken from its budget."""

    def __init__(self, router: "DynamicQueueRouter", priority: int, request_id: str) -> None:
        self.router = router
        self._priority = priority
        self._request_id = request_id
        self._deadline: Final[TotalDeadline | None] = current_total_deadline()
        self._remaining_s = router.settings.queue_timeout
        self._waits = 0

    def _budget_s(self) -> float:
        deadline: Final = self._deadline or current_total_deadline()
        left: Final = None if deadline is None else deadline.remaining()
        return self._remaining_s if left is None else min(self._remaining_s, left)

    async def admit(self, capacity: str, model: str, cooldown_left_s: Callable[[], Awaitable[float]]) -> Admission:
        """Raises the public error of a refused wait. ``cooldown_left_s`` keeps the wait queued, in priority order,
        until the deployment's own cooldown ends."""
        queue: Final = self.router.queue()
        if isinstance(queue, CoordinationUnavailable):
            raise rejection_error(queue, model)
        self._waits += 1
        request: Final = WaitRequest(capacity, f"{self._request_id}:{self._waits}", self._priority)
        budget: Final = self._budget_s()
        admitted: Final = await queue.admit(request, budget, cooldown_left_s)
        if not isinstance(admitted, Admitted):
            raise rejection_error(admitted, model)
        if not isinstance(admitted.grant, PassThrough):
            self._remaining_s = max(self._remaining_s - (budget - admitted.remaining_s), 0.0)
        held: Final = contextlib.AsyncExitStack()
        attempt: Final = await held.enter_async_context(queue.attempt(request, admitted))
        return Admission(attempt, held, self.router)


_CURRENT: Final[ContextVar[RequestQueue | None]] = ContextVar("dynamic_queue_request", default=None)


def current_request_queue() -> RequestQueue | None:
    return _CURRENT.get()


@contextlib.contextmanager
def bound_request_queue(request_queue: RequestQueue | None) -> Generator[None, None, None]:
    token: Final = _CURRENT.set(request_queue)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class DynamicQueueRouter:
    """Reads the router's Redis cache lazily, since the proxy attaches it after the router is built; with none, every
    wait is refused rather than coordinated in one process."""

    def __init__(
        self,
        settings: DynamicQueueSettings,
        redis: Callable[[], ScriptHost | None],
        timer: Timer | None = None,
    ) -> None:
        self.settings = settings
        self.queue_settings: Final = settings.queue_settings()
        self.timer: Final[Timer] = timer or MonotonicTimer()
        self._redis = redis
        self._triggers: Final = frozenset(settings.trigger_status_codes)
        self._built: DynamicQueue | None = None
        self._backend: _Backend | None = None

    def queue(self) -> DynamicQueue | CoordinationUnavailable:
        if self._built is not None:
            return self._built
        backend: Final = self._build_backend()
        if isinstance(backend, CoordinationUnavailable):
            return backend
        self._backend = backend
        self._built = DynamicQueue(backend, self.timer)
        return self._built

    def _build_backend(self) -> _Backend | CoordinationUnavailable:
        if self.settings.coordination == "local":
            return LocalQueueBackend(self.queue_settings, clock=self.timer.now)
        redis: Final = self._redis()
        if redis is None:
            return CoordinationUnavailable(reason=REDIS_NOT_CONFIGURED)
        return RedisQueueBackend(redis, self.queue_settings)

    def request(self, priority: int, request_id: str) -> RequestQueue:
        return RequestQueue(self, priority, request_id)

    async def busy(self, capacities: Mapping[str, str]) -> frozenset[str] | CoordinationUnavailable:
        """The deployment ids whose capacity is congested or still admitting its recovered waiters."""
        queue: Final = self.queue()
        if isinstance(queue, CoordinationUnavailable):
            return queue
        backend: Final = self._backend
        if backend is None:
            return CoordinationUnavailable(reason=REDIS_NOT_CONFIGURED)
        distinct: Final = tuple(frozenset(capacities.values()))
        answers: Final = await asyncio.gather(*(self._read_busy(backend, capacity) for capacity in distinct))
        unreadable: Final = next((answer for answer in answers if isinstance(answer, CoordinationUnavailable)), None)
        if unreadable is not None:
            return unreadable
        busy_capacities: Final = frozenset(capacity for capacity, busy in zip(distinct, answers) if busy is True)
        return frozenset(deployment for deployment, capacity in capacities.items() if capacity in busy_capacities)

    async def _read_busy(self, backend: _Backend, capacity: str) -> bool | CoordinationUnavailable:
        bound_s: Final = self.queue_settings.operation_timeout_ms / 1000
        answer: Final = await _within(self.timer, bound_s, backend.busy(capacity))
        return CoordinationUnavailable(reason=OPERATION_TIMED_OUT) if answer is None else answer

    def outcome_of(self, error: BaseException, retry_after_s: float | None) -> Outcome:
        from litellm.litellm_core_utils.exception_mapping_utils import provider_origin_status

        if provider_origin_status(error) not in self._triggers:
            return OtherFailure()
        return MatchingFailure(retry_after_ms=int((retry_after_s or 0.0) * 1000))


def capacity_of(deployment: Mapping[str, object]) -> str | None:
    model_info: Final = as_str_mapping(deployment.get("model_info"))
    if model_info is None:
        return None
    shared: Final = model_info.get(CAPACITY_ID_FIELD)
    if isinstance(shared, str) and shared:
        return f"shared:{shared}"
    own: Final = model_info.get("id")
    return f"deployment:{own}" if isinstance(own, str) and own else None


def _capacities(records: Sequence[Mapping[str, object]]) -> Mapping[str, str]:
    pairs: Final = ((as_str_mapping(record.get("model_info")), capacity_of(record)) for record in records)
    return {
        str(model_info.get("id")): capacity
        for model_info, capacity in pairs
        if model_info is not None and capacity is not None
    }


Records: TypeAlias = Callable[[Sequence[str]], Sequence[Mapping[str, object]]]


async def congested_ids(
    ids: Sequence[str], records: Records, queues_on: Callable[[Mapping[str, object]], bool]
) -> frozenset[str] | CoordinationUnavailable:
    """Never busy for a deployment the request does not queue on, such as one it reaches on its caller's own
    credentials."""
    request_queue: Final = current_request_queue()
    if request_queue is None or not ids:
        return frozenset()
    queued: Final = tuple(record for record in records(ids) if queues_on(record))
    if not queued:
        return frozenset()
    return await request_queue.router.busy(_capacities(queued))


async def prefer_uncongested(
    ids: Sequence[str], records: Records, queues_on: Callable[[Mapping[str, object]], bool]
) -> tuple[str, ...]:
    """Unknown congestion keeps every candidate; the admission that follows refuses the wait then."""
    busy: Final = await congested_ids(ids, records, queues_on)
    if isinstance(busy, CoordinationUnavailable):
        return tuple(ids)
    free: Final = tuple(deployment for deployment in ids if deployment not in busy)
    return free if free else tuple(ids)


__all__ = [
    "CAPACITY_ID_FIELD",
    "Admission",
    "DynamicQueueCoordinationError",
    "DynamicQueueFullError",
    "DynamicQueueRouter",
    "DynamicQueueSettings",
    "DynamicQueueTimeoutError",
    "QueueRejectedError",
    "Records",
    "RequestQueue",
    "StreamReads",
    "bound_request_queue",
    "capacity_of",
    "congested_ids",
    "current_request_queue",
    "parse_dynamic_queue",
    "prefer_uncongested",
    "refuse_local_queue_across_workers",
    "rejection_error",
]
