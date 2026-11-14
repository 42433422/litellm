"""Per-capacity admission for router attempts once a provider reports overload.

A healthy capacity admits every attempt without a queue entry. A matching provider failure congests it: attempts then
wait in priority order (lower first, FIFO within a priority) and one leased probe at a time is admitted after a
backoff. Enough consecutive probe successes of the current generation recover it into a finite snapshot of the waiters
it held, which claim admission strictly in that order while every later arrival waits; once the snapshot is claimed or
gone the capacity is healthy again. Another matching failure folds unclaimed waiters back into the priority queue.
``LocalQueueBackend`` serves one process; ``RedisQueueBackend`` in ``dynamic_queue_redis`` serves many workers.
"""

import asyncio
import bisect
import time
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType, TracebackType
from typing import Final, Protocol, TypeAlias, TypeVar

OPERATION_TIMED_OUT: Final = "operation_timed_out"


@dataclass(frozen=True, slots=True)
class QueueSettings:
    required_successes: int = 2
    initial_backoff_ms: int = 1_000
    max_backoff_ms: int = 30_000
    lease_ms: int = 10_000
    liveness_ms: int = 5_000
    recheck_ms: int = 250
    claim_step_ms: int = 10
    operation_timeout_ms: int = 1_000
    max_waiting: int = 1_000
    state_ttl_ms: int = 600_000
    sweep_limit: int = 64


@dataclass(frozen=True, slots=True)
class PassThrough:
    generation: int


@dataclass(frozen=True, slots=True)
class Probe:
    generation: int
    token: int


@dataclass(frozen=True, slots=True)
class Released:
    generation: int


Grant: TypeAlias = PassThrough | Probe | Released


@dataclass(frozen=True, slots=True)
class Waiting:
    ahead: int | None = None


@dataclass(frozen=True, slots=True)
class Gone:
    pass


@dataclass(frozen=True, slots=True)
class QueueFull:
    pass


@dataclass(frozen=True, slots=True)
class QueueTimeout:
    pass


@dataclass(frozen=True, slots=True)
class CoordinationUnavailable:
    reason: str


Rejection: TypeAlias = QueueTimeout | QueueFull | CoordinationUnavailable
AdmitReply: TypeAlias = Grant | Waiting | Gone | QueueFull | CoordinationUnavailable
PollReply: TypeAlias = Grant | Waiting | Gone | CoordinationUnavailable


@dataclass(frozen=True, slots=True)
class Succeeded:
    pass


@dataclass(frozen=True, slots=True)
class MatchingFailure:
    retry_after_ms: int = 0


@dataclass(frozen=True, slots=True)
class OtherFailure:
    pass


Outcome: TypeAlias = Succeeded | MatchingFailure | OtherFailure


@dataclass(frozen=True, slots=True)
class WaitRequest:
    capacity: str
    waiter_id: str
    priority: int


@dataclass(frozen=True, slots=True)
class Admitted:
    grant: Grant
    remaining_s: float


_R: Final = TypeVar("_R")


class QueueBackend(Protocol):
    @property
    def settings(self) -> QueueSettings: ...

    async def admit(self, request: WaitRequest, budget_ms: int) -> AdmitReply: ...

    async def poll(self, capacity: str, waiter_id: str) -> PollReply: ...

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable: ...

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable: ...

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable: ...


class Timer(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicTimer:
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


async def _reap(*tasks: "asyncio.Task[object]") -> None:
    unfinished: Final = tuple(task for task in tasks if not task.done())
    for task in unfinished:
        _ = task.cancel()
    if unfinished:
        _ = await asyncio.wait(unfinished)


async def _race(timer: Timer, seconds: float, work: "asyncio.Task[object]") -> bool:
    """Whether ``work`` finished within ``seconds``; unfinished work, or work of a cancelled caller, is cancelled."""
    guard: Final = asyncio.create_task(timer.sleep(seconds))
    try:
        _ = await asyncio.wait((work, guard), return_when=asyncio.FIRST_COMPLETED)
        return work.done()
    finally:
        await _reap(guard, work)


async def _until_done(
    work: "tuple[asyncio.Task[object], ...]", guard: "asyncio.Task[None]", interrupted: bool = False
) -> bool:
    try:
        _ = await asyncio.wait((*work, guard), return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        return await _until_done(work, guard, interrupted=True)
    if guard.done() or all(task.done() for task in work):
        return interrupted
    return await _until_done(work, guard, interrupted)


async def _clean_up(timer: Timer, seconds: float, *work: "asyncio.Task[object]") -> bool:
    """Lets cleanup the caller owns run for up to ``seconds`` however often the caller is cancelled meanwhile, then
    cancels what is left and raises any cancellation it held back. Returns whether all of it finished in time."""
    if not work:
        return True
    guard: Final = asyncio.create_task(timer.sleep(seconds))
    interrupted: Final = await _until_done(work, guard)
    finished: Final = all(task.done() for task in work)
    await _reap(guard, *work)
    if interrupted:
        raise asyncio.CancelledError
    return finished


def _operation_s(backend: QueueBackend) -> float:
    return backend.settings.operation_timeout_ms / 1000


class Attempt:
    """Holds a probe's lease while its provider call runs, and reports the call's outcome once. Leaving without a
    report releases the lease without counting anything, so an abandoned probe is never replayed or credited."""

    def __init__(self, backend: QueueBackend, timer: Timer, request: WaitRequest, grant: Grant) -> None:
        self._backend = backend
        self._timer = timer
        self._request = request
        self._grant = grant
        self._renewal: asyncio.Task[None] | None = None
        self._settled = False

    async def __aenter__(self) -> "Attempt":
        if isinstance(self._grant, Probe):
            self._renewal = asyncio.create_task(self._renew(self._grant.token))
        return self

    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        renewal: Final = self._stop_renewal()
        if self._settled or not isinstance(self._grant, Probe):
            _ = await _clean_up(self._timer, _operation_s(self._backend), *renewal)
            return
        self._settled = True
        release: Final = asyncio.create_task(self._backend.abandon(self._request.capacity, self._request.waiter_id))
        _ = await _clean_up(self._timer, _operation_s(self._backend), release, *renewal)

    async def report(self, outcome: Outcome) -> None | CoordinationUnavailable:
        if self._settled:
            return None
        self._settled = True
        renewal: Final = self._stop_renewal()
        sent: Final = asyncio.create_task(
            self._backend.report(self._request.capacity, self._request.waiter_id, self._grant, outcome)
        )
        if not await _clean_up(self._timer, _operation_s(self._backend), sent, *renewal):
            return CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
        return sent.result()

    async def _renew(self, token: int) -> None:
        interval: Final = self._backend.settings.lease_ms / 3000
        while await self._renewed_after(interval, token):
            pass

    async def _renewed_after(self, interval: float, token: int) -> bool:
        """Renews the lease once ``interval`` passed; False once it is lost. An unreachable backend keeps trying."""
        await self._timer.sleep(interval)
        renewed: Final = await self._backend.renew(self._request.capacity, self._request.waiter_id, token)
        return renewed is not False

    def _stop_renewal(self) -> "tuple[asyncio.Task[None], ...]":
        renewal: Final = self._renewal
        if renewal is None:
            return ()
        self._renewal = None
        _ = renewal.cancel()
        return (renewal,)


class DynamicQueue:
    def __init__(self, backend: QueueBackend, timer: Timer | None = None) -> None:
        self._backend = backend
        self._timer = timer or MonotonicTimer()

    async def admit(self, request: WaitRequest, budget_s: float) -> Admitted | Rejection:
        """Wait for ``request``'s turn on its capacity for at most ``budget_s``, the queue time the request has left.
        ``Admitted.remaining_s`` is what a later wait of the same request may still spend. A healthy capacity admits
        even with no budget left; a congested one never grants past the budget."""
        deadline: Final = self._timer.now() + max(budget_s, 0.0)
        try:
            settled: Final = await self._settle(request, deadline)
        except BaseException:
            await self._abandon(request)
            raise
        if isinstance(settled, (QueueTimeout, CoordinationUnavailable)):
            await self._abandon(request)
        if isinstance(settled, (QueueTimeout, QueueFull, CoordinationUnavailable)):
            return settled
        return Admitted(grant=settled, remaining_s=max(deadline - self._timer.now(), 0.0))

    def attempt(self, request: WaitRequest, admitted: Admitted) -> Attempt:
        return Attempt(self._backend, self._timer, request, admitted.grant)

    def _delay(self, waiting: Waiting) -> float:
        settings: Final = self._backend.settings
        if waiting.ahead is None:
            return settings.recheck_ms / 1000
        return min(settings.recheck_ms, settings.claim_step_ms * (waiting.ahead + 1)) / 1000

    async def _within(self, call: Coroutine[object, object, _R], seconds: float) -> _R | None:
        work: Final = asyncio.create_task(call)
        return work.result() if await _race(self._timer, seconds, work) else None

    def _in_time(self, grant: Grant | QueueFull | CoordinationUnavailable, deadline: float) -> Grant | Rejection:
        late: Final = isinstance(grant, (Probe, Released)) and self._timer.now() >= deadline
        return QueueTimeout() if late else grant

    async def _settle(self, request: WaitRequest, deadline: float) -> Grant | Rejection:
        budget_ms: Final = max(int((deadline - self._timer.now()) * 1000), 0)
        first: Final = await self._within(self._backend.admit(request, budget_ms), _operation_s(self._backend))
        if first is None:
            return CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
        if isinstance(first, Gone):
            return QueueTimeout()
        if isinstance(first, Waiting):
            return await self._wait(request, deadline, first)
        return self._in_time(first, deadline)

    async def _wait(self, request: WaitRequest, deadline: float, waiting: Waiting) -> Grant | Rejection:
        reply: Grant | Rejection | Waiting = waiting  # rebind-ok: each poll's reply decides whether to wait on
        while isinstance(reply, Waiting):
            reply = await self._recheck(request, deadline, reply)
        return reply

    async def _recheck(self, request: WaitRequest, deadline: float, waiting: Waiting) -> Grant | Rejection | Waiting:
        """One recheck: sleeps as long as ``waiting`` asks, within the deadline, then polls once."""
        remaining: Final = deadline - self._timer.now()
        if remaining <= 0:
            return QueueTimeout()
        await self._timer.sleep(min(self._delay(waiting), remaining))
        left: Final = deadline - self._timer.now()
        if left <= 0:
            return QueueTimeout()
        operation_s: Final = _operation_s(self._backend)
        polled: Final = await self._within(
            self._backend.poll(request.capacity, request.waiter_id), min(operation_s, left)
        )
        if polled is None:
            return QueueTimeout() if left < operation_s else CoordinationUnavailable(reason=OPERATION_TIMED_OUT)
        if isinstance(polled, Gone):
            return QueueTimeout()
        if isinstance(polled, Waiting):
            return polled
        return self._in_time(polled, deadline)

    async def _abandon(self, request: WaitRequest) -> None:
        release: Final = asyncio.create_task(self._backend.abandon(request.capacity, request.waiter_id))
        _ = await _clean_up(self._timer, _operation_s(self._backend), release)


@dataclass(frozen=True, slots=True)
class _Waiter:
    waiter_id: str
    priority: int
    seq: int
    deadline_ms: int


@dataclass(frozen=True, slots=True)
class _Lease:
    waiter: _Waiter
    token: int
    until_ms: int
    claimed: bool


@dataclass(frozen=True, slots=True)
class _Capacity:
    congested: bool = False
    generation: int = 0
    successes: int = 0
    backoff_ms: int = 0
    next_probe_ms: int = 0
    seq: int = 0
    tokens: int = 0
    lease: _Lease | None = None
    queue: tuple[_Waiter, ...] = ()
    snapshot: tuple[_Waiter, ...] = ()

    @property
    def healthy(self) -> bool:
        return not self.congested and not self.snapshot


def _order(waiter: _Waiter) -> tuple[int, int]:
    return (waiter.priority, waiter.seq)


def _queued(queue: tuple[_Waiter, ...], waiter: _Waiter) -> tuple[_Waiter, ...]:
    at: Final = bisect.bisect_right(queue, _order(waiter), key=_order)
    return (*queue[:at], waiter, *queue[at:])


def _live(waiters: tuple[_Waiter, ...], now: int) -> tuple[_Waiter, ...]:
    return tuple(waiter for waiter in waiters if waiter.deadline_ms > now)


def _without(waiters: tuple[_Waiter, ...], waiter_id: str) -> tuple[_Waiter, ...]:
    return tuple(waiter for waiter in waiters if waiter.waiter_id != waiter_id)


def _unclaimed_back(capacity: _Capacity, now: int) -> tuple[_Waiter, ...]:
    """The queue with an unclaimed probe's waiter back in its place, keeping its priority, sequence and deadline."""
    lease: Final = capacity.lease
    if lease is None or lease.claimed or lease.waiter.deadline_ms <= now:
        return capacity.queue
    return _queued(capacity.queue, lease.waiter)


def _sweep(capacity: _Capacity, now: int) -> _Capacity:
    lease: Final = capacity.lease
    expired_lease: Final = lease is not None and lease.until_ms <= now
    dead_owner: Final = lease is not None and not lease.claimed and lease.waiter.deadline_ms <= now
    if expired_lease or dead_owner:
        return _sweep(replace(capacity, queue=_unclaimed_back(capacity, now), lease=None), now)
    return replace(capacity, queue=_live(capacity.queue, now), snapshot=_live(capacity.snapshot, now))


def _grant_probe(capacity: _Capacity, now: int, settings: QueueSettings) -> _Capacity:
    if not capacity.congested or capacity.lease is not None or now < capacity.next_probe_ms or not capacity.queue:
        return capacity
    token: Final = capacity.tokens + 1
    return replace(
        capacity,
        tokens=token,
        lease=_Lease(waiter=capacity.queue[0], token=token, until_ms=now + settings.lease_ms, claimed=False),
        queue=capacity.queue[1:],
    )


def _claim_probe(capacity: _Capacity, waiter_id: str) -> tuple[_Capacity, Probe] | None:
    lease: Final = capacity.lease
    if lease is None or lease.claimed or lease.waiter.waiter_id != waiter_id:
        return None
    return replace(capacity, lease=replace(lease, claimed=True)), Probe(capacity.generation, lease.token)


def _congest(
    capacity: _Capacity, now: int, settings: QueueSettings, retry_after_ms: int, failed_probe: bool
) -> _Capacity:
    backoff: Final = (
        settings.initial_backoff_ms
        if not capacity.congested
        else min(capacity.backoff_ms * 2, settings.max_backoff_ms)
        if failed_probe
        else capacity.backoff_ms
    )
    requeued: Final = _unclaimed_back(capacity, now)
    return replace(
        capacity,
        congested=True,
        generation=capacity.generation + 1,
        successes=0,
        backoff_ms=backoff,
        lease=None,
        queue=tuple(sorted((*requeued, *capacity.snapshot), key=_order)),
        snapshot=(),
        next_probe_ms=max(capacity.next_probe_ms, now + max(backoff, retry_after_ms)),
    )


def _recover(capacity: _Capacity) -> _Capacity:
    return replace(
        capacity,
        congested=False,
        generation=capacity.generation + 1,
        successes=0,
        backoff_ms=0,
        next_probe_ms=0,
        lease=None,
        queue=(),
        snapshot=capacity.queue,
    )


def _admit(
    capacity: _Capacity, now: int, settings: QueueSettings, request: WaitRequest, budget_ms: int
) -> tuple[_Capacity, AdmitReply]:
    swept: Final = _sweep(capacity, now)
    if swept.healthy:
        return swept, PassThrough(swept.generation)
    if budget_ms <= 0:
        return swept, Gone()
    if len(swept.queue) + len(swept.snapshot) >= settings.max_waiting:
        return swept, QueueFull()
    waiter: Final = _Waiter(request.waiter_id, request.priority, swept.seq + 1, now + budget_ms)
    queued: Final = replace(swept, seq=waiter.seq, queue=_queued(swept.queue, waiter))
    if not queued.congested:
        return queued, Waiting(ahead=len(queued.snapshot))
    return _settle_congested(_grant_probe(queued, now, settings), request.waiter_id)


def _settle_congested(capacity: _Capacity, waiter_id: str) -> tuple[_Capacity, Probe | Waiting]:
    claimed: Final = _claim_probe(capacity, waiter_id)
    return (capacity, Waiting()) if claimed is None else claimed


def _poll(capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str) -> tuple[_Capacity, PollReply]:
    swept: Final = _sweep(capacity, now)
    claimed: Final = _claim_probe(swept, waiter_id)
    if claimed is not None:
        return claimed
    position: Final = next((at for at, waiter in enumerate(swept.snapshot) if waiter.waiter_id == waiter_id), None)
    if position == 0:
        return replace(swept, snapshot=swept.snapshot[1:]), Released(swept.generation)
    if position is not None:
        return swept, Waiting(ahead=position)
    if all(waiter.waiter_id != waiter_id for waiter in swept.queue):
        return swept, Gone()
    if swept.healthy:
        return replace(swept, queue=_without(swept.queue, waiter_id)), PassThrough(swept.generation)
    if not swept.congested:
        return swept, Waiting(ahead=len(swept.snapshot))
    return _settle_congested(_grant_probe(swept, now, settings), waiter_id)


def _renew(
    capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str, token: int
) -> tuple[_Capacity, bool]:
    swept: Final = _sweep(capacity, now)
    lease: Final = swept.lease
    if lease is None or not lease.claimed or lease.waiter.waiter_id != waiter_id or lease.token != token:
        return swept, False
    return replace(swept, lease=replace(lease, until_ms=now + settings.lease_ms)), True


def _report(
    capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str, grant: Grant, outcome: Outcome
) -> _Capacity:
    swept: Final = _sweep(capacity, now)
    lease: Final = swept.lease
    active_probe: Final = (
        isinstance(grant, Probe)
        and lease is not None
        and lease.claimed
        and lease.waiter.waiter_id == waiter_id
        and lease.token == grant.token
        and swept.generation == grant.generation
    )
    if isinstance(outcome, MatchingFailure):
        return _congest(swept, now, settings, outcome.retry_after_ms, active_probe)
    if not active_probe:
        return swept
    if isinstance(outcome, OtherFailure):
        return _grant_probe(replace(swept, lease=None, successes=0), now, settings)
    credited: Final = replace(swept, lease=None, successes=swept.successes + 1)
    if credited.successes >= settings.required_successes:
        return _recover(credited)
    return _grant_probe(credited, now, settings)


def _abandon(capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str) -> _Capacity:
    swept: Final = _sweep(capacity, now)
    lease: Final = swept.lease
    remaining: Final = replace(
        swept,
        lease=None if lease is not None and lease.waiter.waiter_id == waiter_id else lease,
        queue=_without(swept.queue, waiter_id),
        snapshot=_without(swept.snapshot, waiter_id),
    )
    return _grant_probe(remaining, now, settings)


class LocalQueueBackend:
    """Coordinates the attempts of one process. Every step runs without awaiting, so each is atomic on the loop."""

    def __init__(self, settings: QueueSettings | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings or QueueSettings()
        self._clock = clock
        self._capacities: Mapping[str, _Capacity] = MappingProxyType({})

    @property
    def settings(self) -> QueueSettings:
        return self._settings

    def _step(self, capacity: str, step: Callable[[_Capacity, int], tuple[_Capacity, _R]]) -> _R:
        updated, reply = step(self._capacities.get(capacity, _Capacity()), int(self._clock() * 1000))
        self._capacities = MappingProxyType({**self._capacities, capacity: updated})
        return reply

    async def admit(self, request: WaitRequest, budget_ms: int) -> AdmitReply:
        return self._step(request.capacity, lambda state, now: _admit(state, now, self._settings, request, budget_ms))

    async def poll(self, capacity: str, waiter_id: str) -> PollReply:
        return self._step(capacity, lambda state, now: _poll(state, now, self._settings, waiter_id))

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool:
        return self._step(capacity, lambda state, now: _renew(state, now, self._settings, waiter_id, token))

    async def report(self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome) -> None:
        self._step(capacity, lambda state, now: (_report(state, now, self._settings, waiter_id, grant, outcome), None))

    async def abandon(self, capacity: str, waiter_id: str) -> None:
        self._step(capacity, lambda state, now: (_abandon(state, now, self._settings, waiter_id), None))
