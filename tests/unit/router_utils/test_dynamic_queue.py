import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    Admitted,
    CoordinationUnavailable,
    DynamicQueue,
    Grant,
    LocalQueueBackend,
    MatchingFailure,
    Outcome,
    OtherFailure,
    PassThrough,
    PollReply,
    Probe,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Rejection,
    Released,
    Succeeded,
    Waiting,
    WaitRequest,
)

CAPACITY: Final = "deployment-a"
FAR: Final = 1_000_000


class ManualTimer:
    """Time moves only when the test advances it, and a test can wait until a task is parked on a sleep."""

    def __init__(self) -> None:
        self.at = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []
        self._changed = asyncio.Condition()

    def now(self) -> float:
        return self.at

    async def sleep(self, seconds: float) -> None:
        wake: Final = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.at + seconds, wake))
        async with self._changed:
            self._changed.notify_all()
        await wake

    def _pending(self) -> int:
        return sum(1 for _, wake in self._sleepers if not wake.done())

    async def _sleeping(self, count: int) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._pending() >= count)

    async def parked(self, task: "asyncio.Task[Admitted | Rejection] | None" = None, count: int = 1) -> None:
        sleeping: Final = asyncio.create_task(self._sleeping(count))
        waited: Final = {sleeping} if task is None else {sleeping, task}
        await asyncio.wait(waited, return_when=asyncio.FIRST_COMPLETED)
        sleeping.cancel()

    def advance(self, seconds: float) -> None:
        self.at += seconds
        for due, wake in self._sleepers:
            if due <= self.at and not wake.done():
                wake.set_result(None)
        self._sleepers = [(due, wake) for due, wake in self._sleepers if not wake.done()]


def _backend(timer: ManualTimer, **overrides: int) -> LocalQueueBackend:
    return LocalQueueBackend(QueueSettings(**overrides), clock=timer.now)


async def _congest(backend: LocalQueueBackend, retry_after_ms: int = 0, capacity: str = CAPACITY) -> None:
    await backend.report(capacity, "in-flight", PassThrough(0), MatchingFailure(retry_after_ms))


def _request(waiter_id: str, priority: int = 0, capacity: str = CAPACITY) -> WaitRequest:
    return WaitRequest(capacity=capacity, waiter_id=waiter_id, priority=priority)


async def _enqueue(backend: LocalQueueBackend, *waiters: tuple[str, int]) -> None:
    replies: Final = [await backend.admit(_request(waiter_id, priority), FAR) for waiter_id, priority in waiters]
    assert all(isinstance(reply, Waiting) for reply in replies)


async def _polls(backend: LocalQueueBackend, waiter_ids: Sequence[str]) -> dict[str, PollReply]:
    return {waiter_id: await backend.poll(CAPACITY, waiter_id) for waiter_id in waiter_ids}


async def _probed(backend: LocalQueueBackend, waiter_ids: Sequence[str]) -> tuple[str, Probe]:
    replies: Final = await _polls(backend, waiter_ids)
    probes: Final = [(waiter_id, reply) for waiter_id, reply in replies.items() if isinstance(reply, Probe)]
    assert len(probes) == 1, replies
    assert all(isinstance(reply, (Probe, Waiting)) for reply in replies.values()), replies
    return probes[0]


async def _run(timer: ManualTimer, task: "asyncio.Task[Admitted | Rejection]", step: float) -> Admitted | Rejection:
    while True:
        await timer.parked(task)
        if task.done():
            return task.result()
        timer.advance(step)


async def test_a_healthy_capacity_admits_every_attempt_without_holding_a_queue_place() -> None:
    backend: Final = _backend(ManualTimer(), max_waiting=1)

    replies: Final = [await backend.admit(_request(f"w{index}"), FAR) for index in range(3)]

    assert all(isinstance(reply, PassThrough) for reply in replies)


@pytest.mark.parametrize(
    ("outcome", "congested"),
    [(MatchingFailure(), True), (OtherFailure(), False), (Succeeded(), False)],
)
async def test_only_a_matching_failure_congests_a_healthy_capacity(outcome: Outcome, congested: bool) -> None:
    backend: Final = _backend(ManualTimer())

    await backend.report(CAPACITY, "in-flight", PassThrough(0), outcome)

    reply: Final = await backend.admit(_request("next"), FAR)
    assert isinstance(reply, Waiting) is congested
    assert isinstance(reply, PassThrough) is not congested


async def test_congestion_stays_on_the_capacity_that_failed() -> None:
    backend: Final = _backend(ManualTimer())
    await _congest(backend, capacity="shared-pool")

    assert isinstance(await backend.admit(_request("a", capacity="shared-pool"), FAR), Waiting)
    assert isinstance(await backend.admit(_request("b", capacity="other-pool"), FAR), PassThrough)


async def test_probes_go_in_priority_order_and_first_in_first_out_within_a_priority() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=100)
    await _congest(backend)
    arrivals: Final = (
        ("five-first", 5),
        ("minus-three-first", -3),
        ("zero", 0),
        ("five-second", 5),
        ("huge", 10**30),
        ("minus-three-second", -3),
        ("minus-huge", -(10**30)),
    )
    await _enqueue(backend, *arrivals)
    timer.advance(1.0)

    order: list[str] = []
    remaining: list[str] = [waiter_id for waiter_id, _ in arrivals]
    while remaining:
        waiter_id, probe = await _probed(backend, remaining)
        await backend.report(CAPACITY, waiter_id, probe, OtherFailure())
        order.append(waiter_id)
        remaining.remove(waiter_id)

    assert order == [
        "minus-huge",
        "minus-three-first",
        "minus-three-second",
        "zero",
        "five-first",
        "five-second",
        "huge",
    ]


async def test_probe_backoff_doubles_on_each_failed_probe_up_to_the_cap_and_honors_a_longer_retry_after() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, initial_backoff_ms=1_000, max_backoff_ms=3_000)
    await _congest(backend)
    await _enqueue(backend, ("a", 0), ("b", 0), ("c", 0), ("d", 0))
    probe_times_and_retry_after: Final = ((1.0, 0), (3.0, 0), (6.0, 10_000), (16.0, 0))

    for (due, retry_after_ms), waiter_id in zip(probe_times_and_retry_after, "abcd"):
        timer.at = due - 0.001
        assert isinstance(await backend.poll(CAPACITY, waiter_id), Waiting)
        timer.at = due
        probe = await backend.poll(CAPACITY, waiter_id)
        assert isinstance(probe, Probe)
        await backend.report(CAPACITY, waiter_id, probe, MatchingFailure(retry_after_ms))


async def test_recovery_needs_consecutive_probe_successes_then_admits_every_waiter_ahead_of_later_arrivals() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=2)
    await _congest(backend)
    await _enqueue(backend, ("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4), ("f", 4))
    timer.advance(1.0)
    waiters: Final = ["a", "b", "c", "d", "e", "f"]

    for waiter_id, outcome in (("a", Succeeded()), ("b", OtherFailure()), ("c", Succeeded())):
        probed, probe = await _probed(backend, waiters)
        assert probed == waiter_id
        await backend.report(CAPACITY, waiter_id, probe, outcome)
        waiters.remove(waiter_id)
    probed, probe = await _probed(backend, waiters)
    assert probed == "d"
    await backend.report(CAPACITY, "d", probe, Succeeded())
    arrival: Final = await backend.admit(_request("after-recovery", priority=-100), FAR)

    released: Final = await _polls(backend, ["e", "f"])
    assert isinstance(arrival, PassThrough)
    assert isinstance(released["e"], Released) and isinstance(released["f"], Released)
    assert (released["e"].order, released["f"].order) == (1, 2)
    assert released["e"].generation == released["f"].generation == arrival.generation


async def test_a_probe_success_from_a_superseded_generation_does_not_recover() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    await _congest(backend)
    await _enqueue(backend, ("a", 0))
    timer.advance(1.0)
    stale: Final = await backend.poll(CAPACITY, "a")
    assert isinstance(stale, Probe)

    await _congest(backend)
    await backend.report(CAPACITY, "a", stale, Succeeded())

    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)
    timer.advance(1.0)
    fresh: Final = await backend.poll(CAPACITY, "b")
    assert isinstance(fresh, Probe) and fresh.generation > stale.generation
    await backend.report(CAPACITY, "b", fresh, Succeeded())
    assert isinstance(await backend.admit(_request("c"), FAR), PassThrough)


async def test_an_unrenewed_probe_lease_expires_so_the_next_waiter_probes_and_the_late_result_is_ignored() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1, lease_ms=3_000)
    await _congest(backend)
    await _enqueue(backend, ("lost-worker", 0), ("b", 0))
    timer.advance(1.0)
    abandoned: Final = await backend.poll(CAPACITY, "lost-worker")
    assert isinstance(abandoned, Probe)

    timer.at = 3.999
    assert isinstance(await backend.poll(CAPACITY, "b"), Waiting)
    timer.at = 4.0
    takeover: Final = await backend.poll(CAPACITY, "b")
    await backend.report(CAPACITY, "lost-worker", abandoned, Succeeded())

    assert isinstance(takeover, Probe)
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_a_full_queue_refuses_new_waiters_until_one_leaves() -> None:
    backend: Final = _backend(ManualTimer(), max_waiting=2)
    await _congest(backend, retry_after_ms=FAR)
    await _enqueue(backend, ("a", 0), ("b", 0))

    assert isinstance(await backend.admit(_request("c"), FAR), QueueFull)
    await backend.abandon(CAPACITY, "a")
    assert isinstance(await backend.admit(_request("c"), FAR), Waiting)


async def test_a_waiter_times_out_on_its_budget_and_gives_its_place_back() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, max_waiting=1, recheck_ms=250)
    await _congest(backend, retry_after_ms=FAR)
    queue: Final = DynamicQueue(backend, timer)

    result: Final = await _run(timer, asyncio.create_task(queue.admit(_request("a"), 1.0)), 0.25)

    assert isinstance(result, QueueTimeout)
    assert timer.at == 1.0
    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)


async def test_an_admitted_waiter_keeps_the_rest_of_its_budget_for_a_later_wait() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, recheck_ms=250, initial_backoff_ms=1_000)
    await _congest(backend)
    queue: Final = DynamicQueue(backend, timer)

    result: Final = await _run(timer, asyncio.create_task(queue.admit(_request("a"), 2.5)), 0.25)

    assert isinstance(result, Admitted) and isinstance(result.grant, Probe)
    assert result.remaining_s == 1.5


async def test_a_healthy_capacity_admits_even_with_no_budget_left() -> None:
    timer: Final = ManualTimer()
    queue: Final = DynamicQueue(_backend(timer), timer)

    result: Final = await queue.admit(_request("a"), 0.0)

    assert isinstance(result, Admitted) and isinstance(result.grant, PassThrough)


async def test_cancelling_a_waiter_removes_it_from_the_queue() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, max_waiting=1)
    await _congest(backend, retry_after_ms=FAR)
    waiter: Final = asyncio.create_task(DynamicQueue(backend, timer).admit(_request("a"), 30.0))
    await timer.parked(waiter)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert isinstance(await backend.admit(_request("b"), FAR), Waiting)


class StallsAfterEnqueue:
    """Enqueues for real, then never answers, like a Redis reply lost after the script ran."""

    def __init__(self, inner: LocalQueueBackend) -> None:
        self.inner = inner
        self.enqueued = asyncio.Event()

    @property
    def settings(self) -> QueueSettings:
        return self.inner.settings

    async def admit(self, request: WaitRequest, budget_ms: int) -> AdmitReply:
        _ = await self.inner.admit(request, budget_ms)
        self.enqueued.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def poll(self, capacity: str, waiter_id: str) -> PollReply:
        return await self.inner.poll(capacity, waiter_id)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        return await self.inner.renew(capacity, waiter_id, token)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return await self.inner.report(capacity, waiter_id, grant, outcome)

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        return await self.inner.abandon(capacity, waiter_id)


async def test_cancelling_while_the_enqueue_reply_is_in_flight_still_removes_the_waiter() -> None:
    timer: Final = ManualTimer()
    inner: Final = _backend(timer, max_waiting=1)
    await _congest(inner, retry_after_ms=FAR)
    stalling: Final = StallsAfterEnqueue(inner)
    waiter: Final = asyncio.create_task(DynamicQueue(stalling, timer).admit(_request("a"), 30.0))
    await stalling.enqueued.wait()

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert isinstance(await inner.admit(_request("b"), FAR), Waiting)


async def _probe_attempt(timer: ManualTimer, backend: LocalQueueBackend) -> tuple[DynamicQueue, Admitted]:
    await _congest(backend)
    queue: Final = DynamicQueue(backend, timer)
    admitted: Final = await _run(timer, asyncio.create_task(queue.admit(_request("prober"), 100.0)), 0.25)
    assert isinstance(admitted, Admitted) and isinstance(admitted.grant, Probe)
    return queue, admitted


async def test_a_running_probe_renews_its_lease_and_leaving_unreported_hands_the_probe_on() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, lease_ms=3_000, recheck_ms=250)
    queue, admitted = await _probe_attempt(timer, backend)
    await _enqueue(backend, ("next", 0))

    async with queue.attempt(_request("prober"), admitted):
        for _ in range(5):
            await timer.parked()
            timer.advance(1.0)
        await timer.parked()
        held: Final = await backend.poll(CAPACITY, "next")

    handed_on: Final = await backend.poll(CAPACITY, "next")
    assert isinstance(held, Waiting)
    assert isinstance(handed_on, Probe)


async def test_a_reported_probe_success_recovers_the_capacity_for_the_waiters_behind_it() -> None:
    timer: Final = ManualTimer()
    backend: Final = _backend(timer, required_successes=1)
    queue, admitted = await _probe_attempt(timer, backend)
    await _enqueue(backend, ("behind", 0))

    async with queue.attempt(_request("prober"), admitted) as attempt:
        assert await attempt.report(Succeeded()) is None

    assert isinstance(await backend.poll(CAPACITY, "behind"), Released)
