"""The router's side of ``dynamic_queue``: the ``router_settings.dynamic_queue`` block, the queue time each request
owns across its attempts, per-attempt admission, and the public errors a refused wait raises.

Enabled, every async chat or text completion attempt on a deployment the router owns waits for its capacity's turn,
and its outcome is reported back. Only a failure whose provider answered with one of ``trigger_status_codes`` congests
a capacity. A refused wait ends the request with no retry, fallback or cooldown.
"""

import asyncio
import contextlib
from collections.abc import Callable, Generator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import assert_never

from litellm.exceptions import ServiceUnavailableError, Timeout
from litellm.litellm_core_utils.total_deadline import current_total_deadline
from litellm.router_utils.dynamic_queue import (
    Admitted,
    Attempt,
    CoordinationUnavailable,
    DynamicQueue,
    LocalQueueBackend,
    MatchingFailure,
    OtherFailure,
    Outcome,
    PassThrough,
    MonotonicTimer,
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


class DynamicQueueSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    trigger_status_codes: tuple[int, ...] = (429, 529)
    queue_timeout: float = Field(default=30.0, gt=0)
    max_waiting: int = Field(default=1_000, gt=0)
    coordination: Literal["redis", "local"] = "redis"


_SETTINGS: Final = TypeAdapter(DynamicQueueSettings | None)


def parse_dynamic_queue(value: object, default_priority: int | None) -> DynamicQueueSettings | None:
    """The ``dynamic_queue`` router setting, or None when it is unset or off. Raises on an invalid block, and when the
    queue is enabled without the ``default_priority`` every unlabelled request queues at."""
    settings: Final = _SETTINGS.validate_python(value)
    if settings is None or not settings.enabled:
        return None
    if default_priority is None:
        raise ValueError("router_settings.dynamic_queue.enabled requires router_settings.default_priority")
    return settings


def refuse_local_queue_across_workers(value: object, num_workers: str) -> None:
    """A proxy running several workers cannot share a process-local queue, so it refuses to start with one."""
    settings: Final = _SETTINGS.validate_python(value)
    if settings is not None and settings.enabled and settings.coordination == "local" and num_workers != "1":
        raise ValueError(
            f"router_settings.dynamic_queue.coordination 'local' serves one process, not {num_workers} workers; "
            "use 'redis'"
        )


class QueueRejectedError(Exception):
    """A request the dynamic queue refused. Each public error below is also this class, which the router reads to end
    the request without a retry, fallback or cooldown."""


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


@dataclass(frozen=True, slots=True)
class Admission:
    """An admitted attempt, held from before its provider call until its outcome is known: the call's answer, or its
    stream's end. ``finish`` reports the outcome, if any, and releases the attempt exactly once."""

    attempt: Attempt
    held: contextlib.AsyncExitStack
    router: "DynamicQueueRouter"

    async def finish(self, outcome: Outcome | None) -> None:
        if outcome is not None:
            _ = await self.attempt.report(outcome)
        await self.held.aclose()


class _Backend(QueueBackend, Protocol):
    async def busy(self, capacity: str) -> bool | CoordinationUnavailable: ...


class RequestQueue:
    """The queue state one request owns across every provider attempt it makes, its stream's recovery included: its
    priority, the queue time it has left, and a new waiter id for each wait. Only time spent actually waiting is
    taken from the budget; an attempt a healthy capacity admits at once costs none of it."""

    def __init__(self, router: "DynamicQueueRouter", priority: int, request_id: str) -> None:
        self.router = router
        self._priority = priority
        self._request_id = request_id
        self._remaining_s = router.settings.queue_timeout
        self._waits = 0

    async def admit(self, capacity: str, model: str) -> Admission:
        """Waits for ``capacity``'s turn and returns the admitted attempt. Raises the public error of a refused wait."""
        queue: Final = self.router.queue()
        if isinstance(queue, CoordinationUnavailable):
            raise rejection_error(queue, model)
        self._waits += 1
        request: Final = WaitRequest(capacity, f"{self._request_id}:{self._waits}", self._priority)
        deadline: Final = current_total_deadline()
        left: Final = None if deadline is None else deadline.remaining()
        budget: Final = self._remaining_s if left is None else min(self._remaining_s, left)
        admitted: Final = await queue.admit(request, budget)
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
    """Runs the block as part of ``request_queue``'s request: its routed call, or a later recovery of its stream."""
    token: Final = _CURRENT.set(request_queue)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@dataclass(frozen=True, slots=True)
class _Unbuilt:
    pass


class DynamicQueueRouter:
    """One router's queue. Redis coordination reads the router's Redis cache when the first request needs it, since
    the proxy attaches that cache after the router is built; with none, every wait is refused rather than coordinated
    in one process only."""

    def __init__(
        self,
        settings: DynamicQueueSettings,
        redis: Callable[[], ScriptHost | None],
        cooldown_s: Callable[[], float],
        timer: Timer | None = None,
    ) -> None:
        self.settings = settings
        self._redis = redis
        self._cooldown_s = cooldown_s
        self._timer: Final = timer or MonotonicTimer()
        self._triggers: Final = frozenset(settings.trigger_status_codes)
        self._built: DynamicQueue | _Unbuilt = _Unbuilt()
        self._backend: _Backend | None = None

    def _queue_settings(self) -> QueueSettings:
        return QueueSettings(max_waiting=self.settings.max_waiting)

    def queue(self) -> DynamicQueue | CoordinationUnavailable:
        if isinstance(self._built, DynamicQueue):
            return self._built
        backend: Final = self._build_backend()
        if isinstance(backend, CoordinationUnavailable):
            return backend
        self._backend = backend
        self._built = DynamicQueue(backend, self._timer)
        return self._built

    def _build_backend(self) -> _Backend | CoordinationUnavailable:
        if self.settings.coordination == "local":
            return LocalQueueBackend(self._queue_settings(), clock=self._timer.now)
        redis: Final = self._redis()
        if redis is None:
            return CoordinationUnavailable(reason=REDIS_NOT_CONFIGURED)
        return RedisQueueBackend(redis, self._queue_settings())

    def request(self, priority: int, request_id: str) -> RequestQueue:
        return RequestQueue(self, priority, request_id)

    async def busy(self, capacities: Mapping[str, str]) -> frozenset[str]:
        """The ids in ``capacities`` (deployment id to capacity) whose capacity is congested or still admitting its
        recovered waiters. An unreadable capacity counts as not busy, so routing falls back to what it does today."""
        if isinstance(self.queue(), CoordinationUnavailable) or self._backend is None:
            return frozenset()
        backend: Final = self._backend
        distinct: Final = tuple(frozenset(capacities.values()))
        answers: Final = await asyncio.gather(*(self._read_busy(backend, capacity) for capacity in distinct))
        busy_capacities: Final = frozenset(capacity for capacity, busy in zip(distinct, answers) if busy)
        return frozenset(deployment for deployment, capacity in capacities.items() if capacity in busy_capacities)

    async def _read_busy(self, backend: _Backend, capacity: str) -> bool:
        try:
            answer: Final = await asyncio.wait_for(backend.busy(capacity), backend.settings.operation_timeout_ms / 1000)
        except asyncio.TimeoutError:
            return False
        return answer is True

    def outcome_of(self, error: BaseException, retry_after_s: float | None) -> Outcome:
        """A failure congests only when its provider answered with a trigger status. The probe then waits at least the
        router's cooldown and the provider's ``retry-after``."""
        from litellm.litellm_core_utils.exception_mapping_utils import provider_origin_status

        if provider_origin_status(error) not in self._triggers:
            return OtherFailure()
        wait_s: Final = max(retry_after_s or 0.0, self._cooldown_s())
        return MatchingFailure(retry_after_ms=int(wait_s * 1000))


def _capacities(records: Sequence[Mapping[str, object]]) -> Mapping[str, str]:
    pairs: Final = ((record.get("model_info"), capacity_of(record)) for record in records)
    return {
        str(model_info.get("id")): capacity
        for model_info, capacity in pairs
        if isinstance(model_info, Mapping) and capacity is not None
    }


async def congested_ids(
    ids: Sequence[str], records: Callable[[Sequence[str]], Sequence[Mapping[str, object]]]
) -> frozenset[str]:
    """The deployments among ``ids`` whose capacity the current request's queue holds busy, reading their ``records``
    only then; none when the request does not queue."""
    request_queue: Final = current_request_queue()
    if request_queue is None or not ids:
        return frozenset()
    return await request_queue.router.busy(_capacities(records(ids)))


async def prefer_uncongested(
    ids: Sequence[str], records: Callable[[Sequence[str]], Sequence[Mapping[str, object]]]
) -> tuple[str, ...]:
    """``ids`` less the ones whose capacity is busy, while at least one is not, so a healthy alternative is picked
    first. A pick left with only cooling busy ones keeps them through its dynamic congestion stage instead."""
    busy: Final = await congested_ids(ids, records)
    free: Final = tuple(deployment for deployment in ids if deployment not in busy)
    return free if free else tuple(ids)


def capacity_of(deployment: Mapping[str, object]) -> str | None:
    """The capacity a deployment draws on: its ``model_info.dynamic_queue_capacity_id`` when deployments share one
    quota, else its own id."""
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return None
    shared: Final = model_info.get(CAPACITY_ID_FIELD)
    if isinstance(shared, str) and shared:
        return shared
    own: Final = model_info.get("id")
    return own if isinstance(own, str) and own else None


__all__ = [
    "CAPACITY_ID_FIELD",
    "DynamicQueueCoordinationError",
    "DynamicQueueFullError",
    "DynamicQueueRouter",
    "DynamicQueueSettings",
    "DynamicQueueTimeoutError",
    "QueueRejectedError",
    "Admission",
    "RequestQueue",
    "bound_request_queue",
    "capacity_of",
    "congested_ids",
    "current_request_queue",
    "parse_dynamic_queue",
    "prefer_uncongested",
    "refuse_local_queue_across_workers",
    "rejection_error",
]
