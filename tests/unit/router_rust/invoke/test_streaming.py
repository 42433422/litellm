import asyncio
import contextvars
from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.router_rust.invoke.streaming import StreamAttempt, _areported, _chat_total_tokens
from litellm.router_utils.dynamic_queue import MatchingFailure
from litellm.router_utils.dynamic_queue_router import (
    Admission,
    DynamicQueueRouter,
    DynamicQueueSettings,
    RequestQueue,
    current_request_queue,
)
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _finished, _reap

CAPACITY: Final = "deployment:d1"


async def _no_cooldown() -> float:
    return 0.0


class _ProviderAnswered(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("upstream failure")
        self.message = "upstream failure"
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        self.response = httpx.Response(status_code=status_code, request=self.request, text="upstream failure")


def _provider_error(status_code: int) -> Exception:
    try:
        exception_type(model="m", original_exception=_ProviderAnswered(status_code), custom_llm_provider="openai")
    except Exception as mapped:  # noqa: BLE001  # the mapped error is the value under test
        return mapped
    raise AssertionError("exception_type always raises for a provider error")


class _RecoveringCore:
    """Stands in for the core a stream recovers through, noting which request queue the recovery waited with."""

    def __init__(self) -> None:
        self.recovered_with: list[RequestQueue | None] = []

    async def recover_stream(self, *args: object) -> object:
        self.recovered_with.append(current_request_queue())
        return "recovered"

    async def retry_stream(self, *args: object) -> object:
        self.recovered_with.append(current_request_queue())
        return ("retried", 1, 1)


def _stream_attempt(
    admission: Admission | None, request_queue: RequestQueue | None = None, core: object = None
) -> StreamAttempt:
    return StreamAttempt(
        core=core,  # pyright: ignore[reportArgumentType]  # the queue reporting under test reads no core
        op="acompletion",
        request={},
        kwargs={},
        group="group",
        order=None,
        depth=0,
        level=0,
        deployment_id="d1",
        attempted_retries=0,
        admission=admission,
        request_queue=request_queue,
    )


async def _probe(timer: ManualTimer) -> tuple[DynamicQueueRouter, Admission]:
    """A local queue whose capacity a provider failure congested, and the probe its first waiter was granted."""
    router: Final = DynamicQueueRouter(
        DynamicQueueSettings(enabled=True, coordination="local", recovery_successes=1),
        redis=lambda: None,
        timer=timer,
    )
    healthy: Final = await router.request(priority=0, request_id="congests").admit(CAPACITY, "m", _no_cooldown)
    await healthy.finish(MatchingFailure())
    timer.advance(1.0)
    probe: Final = await router.request(priority=0, request_id="probe").admit(CAPACITY, "m", _no_cooldown)
    return router, probe


async def _busy(router: DynamicQueueRouter) -> bool:
    busy: Final = await router.busy({"d1": CAPACITY})
    assert isinstance(busy, frozenset)
    return "d1" in busy


async def _next_probe_granted_at_once(timer: ManualTimer, router: DynamicQueueRouter) -> bool:
    """Whether another request is handed the probe right away, which it is only once the stream gave its turn back."""
    admission: Final = router.request(priority=0, request_id="next").admit(CAPACITY, "m", _no_cooldown)
    waiter: Final = asyncio.create_task(admission)
    try:
        await timer.parked(waiter)
        granted: Final = waiter.done() and not waiter.cancelled() and waiter.exception() is None
        if granted:
            await waiter.result().finish(None)
        return granted
    finally:
        await _reap(waiter)


async def _chunks(*items: object, then: BaseException | None = None) -> AsyncIterator[object]:
    for item in items:
        yield item
    if then is not None:
        raise then


async def _settled(attempt: StreamAttempt) -> None:
    settling: Final = attempt.settling
    assert settling is not None
    await _finished(settling)


async def test_a_stream_read_to_its_end_counts_as_the_probes_success() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)

    read: Final = [chunk async for chunk in _areported(attempt, _chunks("a", "b"), _chat_total_tokens)]
    await _settled(attempt)

    assert read == ["a", "b"]
    assert await _busy(router) is False


@pytest.mark.parametrize(
    "failure",
    [_provider_error(429), litellm.RateLimitError(message="deployment at capacity", llm_provider="", model="m")],
    ids=["provider_429", "litellm_raised_429"],
)
async def test_a_stream_that_fails_midway_never_counts_as_a_success(failure: Exception) -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)

    with pytest.raises(type(failure)):
        _ = [chunk async for chunk in _areported(attempt, _chunks("a", then=failure), _chat_total_tokens)]
    await _settled(attempt)

    assert await _busy(router) is True


async def test_a_stream_closed_after_some_output_gives_its_turn_back_without_credit() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)
    stream: Final = _areported(attempt, _chunks("a", "b", "c"), _chat_total_tokens)

    assert await stream.__anext__() == "a"
    await stream.aclose()
    await _settled(attempt)

    assert await _busy(router) is True
    assert await _next_probe_granted_at_once(timer, router) is True


async def test_cancelling_the_task_reading_a_stream_gives_its_turn_back_without_credit() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)
    hold: Final = asyncio.Event()
    first_read: Final = asyncio.Event()

    async def held_chunks() -> AsyncIterator[object]:
        yield "a"
        await hold.wait()
        yield "b"

    async def read() -> None:
        async for _ in _areported(attempt, held_chunks(), _chat_total_tokens):
            first_read.set()

    reader: Final = asyncio.create_task(read())
    try:
        await asyncio.wait_for(first_read.wait(), 10)
        _ = reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await _finished(reader)
    finally:
        await _reap(reader)
    await _settled(attempt)

    assert await _busy(router) is True
    assert await _next_probe_granted_at_once(timer, router) is True


async def test_a_stream_nobody_reads_gives_its_turn_back_once_a_lease_passes() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)
    stream: Final = _areported(attempt, _chunks("a"), _chat_total_tokens)
    watch: Final = attempt.watch_unread()
    assert watch is not None
    try:
        await timer.parked(count=2)
        timer.advance(router.queue_settings.lease_ms / 1000)
        await _finished(watch)
        released: Final = await _next_probe_granted_at_once(timer, router)
    finally:
        await _reap(watch)
        await stream.aclose()

    assert released is True


async def test_a_stream_its_reader_started_keeps_its_turn_past_the_unread_lease() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _stream_attempt(probe)
    stream: Final = _areported(attempt, _chunks("a", "b"), _chat_total_tokens)
    watch: Final = attempt.watch_unread()
    assert watch is not None
    try:
        assert await stream.__anext__() == "a"
        await _finished(watch)
        timer.advance(router.queue_settings.lease_ms / 1000)
        held: Final = not await _next_probe_granted_at_once(timer, router)
    finally:
        await _reap(watch)
        await stream.aclose()

    assert held is True


@pytest.mark.parametrize("recovery", ["fallback", "retry"])
async def test_a_stream_recovered_from_another_task_waits_with_its_own_requests_queue(recovery: str) -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(
        DynamicQueueSettings(enabled=True, coordination="local"), redis=lambda: None, timer=timer
    )
    request_queue: Final = router.request(priority=-1, request_id="r")
    core: Final = _RecoveringCore()
    attempt: Final = _stream_attempt(None, request_queue, core)
    recover: Final = attempt.fallback if recovery == "fallback" else attempt.retry

    consumer: Final = contextvars.Context().run(asyncio.ensure_future, recover(RuntimeError("cut")))
    _ = await _finished(consumer)

    assert core.recovered_with == [request_queue]
    assert current_request_queue() is None

