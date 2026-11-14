"""Per-capacity admission for router attempts once a provider reports overload.

A healthy capacity admits every attempt without a queue entry. A matching provider failure congests it: attempts then
wait in priority order (lower first, FIFO within a priority), one leased probe at a time is admitted after a backoff,
and enough consecutive probe successes of the current generation recover it, admitting every waiter at once ahead of
any later arrival. ``LocalQueueBackend`` serves one process; ``RedisQueueBackend`` in ``dynamic_queue_redis`` serves
many workers with the same contract.
"""

import asyncio
import bisect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType, TracebackType
from typing import Final, Protocol, TypeAlias, TypeVar


@dataclass(frozen=True, slots=True)
class QueueSettings:
    required_successes: int = 2
    initial_backoff_ms: int = 1_000
    max_backoff_ms: int = 30_000
    lease_ms: int = 10_000
    liveness_ms: int = 5_000
    recheck_ms: int = 250
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
    order: int


Grant: TypeAlias = PassThrough | Probe | Released


@dataclass(frozen=True, slots=True)
class Waiting:
    pass


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
AdmitReply: TypeAlias = Grant | Waiting | QueueFull | CoordinationUnavailable
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
        await self._stop_renewal()
        if self._settled or not isinstance(self._grant, Probe):
            return
        self._settled = True
        _ = await asyncio.shield(self._backend.abandon(self._request.capacity, self._request.waiter_id))

    async def report(self, outcome: Outcome) -> None | CoordinationUnavailable:
        if self._settled:
            return None
        self._settled = True
        await self._stop_renewal()
        return await asyncio.shield(
            self._backend.report(self._request.capacity, self._request.waiter_id, self._grant, outcome)
        )

    async def _renew(self, token: int) -> None:
        interval: Final = self._backend.settings.lease_ms / 3000
        while True:
            await self._timer.sleep(interval)
            renewed: Final = await self._backend.renew(self._request.capacity, self._request.waiter_id, token)
            if renewed is False:
                return

    async def _stop_renewal(self) -> None:
        renewal: Final = self._renewal
        if renewal is None:
            return
        self._renewal = None
        _ = renewal.cancel()
        _ = await asyncio.gather(renewal, return_exceptions=True)


class DynamicQueue:
    def __init__(self, backend: QueueBackend, timer: Timer | None = None) -> None:
        self._backend = backend
        self._timer = timer or MonotonicTimer()

    async def admit(self, request: WaitRequest, budget_s: float) -> Admitted | Rejection:
        """Wait for ``request``'s turn on its capacity for at most ``budget_s``, the queue time the request has left.
        ``Admitted.remaining_s`` is what a later wait of the same request may still spend."""
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

    async def _settle(self, request: WaitRequest, deadline: float) -> Grant | Rejection:
        recheck: Final = self._backend.settings.recheck_ms / 1000
        budget_ms: Final = max(int((deadline - self._timer.now()) * 1000), 0)
        first: Final = await self._backend.admit(request, budget_ms)
        if not isinstance(first, Waiting):
            return first
        while True:
            remaining: Final = deadline - self._timer.now()
            if remaining <= 0:
                return QueueTimeout()
            await self._timer.sleep(min(recheck, remaining))
            polled: Final = await self._backend.poll(request.capacity, request.waiter_id)
            if isinstance(polled, Gone):
                return QueueTimeout()
            if not isinstance(polled, Waiting):
                return polled

    async def _abandon(self, request: WaitRequest) -> None:
        _ = await asyncio.shield(self._backend.abandon(request.capacity, request.waiter_id))


@dataclass(frozen=True, slots=True)
class _Waiter:
    waiter_id: str
    priority: int
    seq: int
    deadline_ms: int


@dataclass(frozen=True, slots=True)
class _Lease:
    waiter_id: str
    token: int
    until_ms: int


@dataclass(frozen=True, slots=True)
class _Pending:
    grant: Grant
    deadline_ms: int


def _no_pending() -> Mapping[str, _Pending]:
    return MappingProxyType({})


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
    pending: Mapping[str, _Pending] = field(default_factory=_no_pending)


_R = TypeVar("_R")


def _order(waiter: _Waiter) -> tuple[int, int]:
    return (waiter.priority, waiter.seq)


def _without_pending(pending: Mapping[str, _Pending], dropped: frozenset[str]) -> Mapping[str, _Pending]:
    return MappingProxyType({waiter_id: entry for waiter_id, entry in pending.items() if waiter_id not in dropped})


def _sweep(capacity: _Capacity, now: int) -> _Capacity:
    expired: Final = frozenset(waiter_id for waiter_id, entry in capacity.pending.items() if entry.deadline_ms <= now)
    lease: Final = capacity.lease
    lease_alive: Final = lease is not None and lease.until_ms > now and lease.waiter_id not in expired
    return replace(
        capacity,
        lease=lease if lease_alive else None,
        queue=tuple(waiter for waiter in capacity.queue if waiter.deadline_ms > now),
        pending=_without_pending(capacity.pending, expired),
    )


def _grant_probe(capacity: _Capacity, now: int, settings: QueueSettings) -> _Capacity:
    if not capacity.congested or capacity.lease is not None or now < capacity.next_probe_ms or not capacity.queue:
        return capacity
    head: Final = capacity.queue[0]
    token: Final = capacity.tokens + 1
    return replace(
        capacity,
        tokens=token,
        lease=_Lease(waiter_id=head.waiter_id, token=token, until_ms=now + settings.lease_ms),
        queue=capacity.queue[1:],
        pending=MappingProxyType(
            {**capacity.pending, head.waiter_id: _Pending(Probe(capacity.generation, token), head.deadline_ms)}
        ),
    )


def _take(capacity: _Capacity, waiter_id: str) -> tuple[_Capacity, Grant | Waiting]:
    entry: Final = capacity.pending.get(waiter_id)
    if entry is None:
        return capacity, Waiting()
    return replace(capacity, pending=_without_pending(capacity.pending, frozenset((waiter_id,)))), entry.grant


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
    return replace(
        capacity,
        congested=True,
        generation=capacity.generation + 1,
        successes=0,
        backoff_ms=backoff,
        lease=None,
        next_probe_ms=max(capacity.next_probe_ms, now + max(backoff, retry_after_ms)),
    )


def _recover(capacity: _Capacity) -> _Capacity:
    generation: Final = capacity.generation + 1
    released: Final = {
        waiter.waiter_id: _Pending(Released(generation, order), waiter.deadline_ms)
        for order, waiter in enumerate(capacity.queue, start=1)
    }
    return replace(
        capacity,
        congested=False,
        generation=generation,
        successes=0,
        backoff_ms=0,
        next_probe_ms=0,
        lease=None,
        queue=(),
        pending=MappingProxyType({**capacity.pending, **released}),
    )


def _admit(
    capacity: _Capacity, now: int, settings: QueueSettings, request: WaitRequest, budget_ms: int
) -> tuple[_Capacity, AdmitReply]:
    swept: Final = _sweep(capacity, now)
    if not swept.congested:
        return swept, PassThrough(swept.generation)
    if len(swept.queue) >= settings.max_waiting:
        return swept, QueueFull()
    waiter: Final = _Waiter(request.waiter_id, request.priority, swept.seq + 1, now + budget_ms)
    at: Final = bisect.bisect_right(swept.queue, _order(waiter), key=_order)
    queued: Final = replace(swept, seq=waiter.seq, queue=(*swept.queue[:at], waiter, *swept.queue[at:]))
    return _take(_grant_probe(queued, now, settings), request.waiter_id)


def _poll(capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str) -> tuple[_Capacity, PollReply]:
    swept: Final = _sweep(capacity, now)
    if waiter_id not in swept.pending and all(waiter.waiter_id != waiter_id for waiter in swept.queue):
        return swept, Gone()
    return _take(_grant_probe(swept, now, settings), waiter_id)


def _renew(
    capacity: _Capacity, now: int, settings: QueueSettings, waiter_id: str, token: int
) -> tuple[_Capacity, bool]:
    swept: Final = _sweep(capacity, now)
    lease: Final = swept.lease
    if lease is None or lease.waiter_id != waiter_id or lease.token != token:
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
        and lease.waiter_id == waiter_id
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
        lease=None if lease is not None and lease.waiter_id == waiter_id else lease,
        queue=tuple(waiter for waiter in swept.queue if waiter.waiter_id != waiter_id),
        pending=_without_pending(swept.pending, frozenset((waiter_id,))),
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
