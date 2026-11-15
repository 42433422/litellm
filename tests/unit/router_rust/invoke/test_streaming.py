import asyncio
import contextlib
import contextvars
from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.exceptions import MidStreamFallbackError
from litellm.router import router_arguments
from litellm.router_config import build_router_config
from litellm.router_rust.invoke.streaming import (
    StreamAttempt,
    _areported,
    _chat_total_tokens,
    _FirstChunk,
    _raise_fallback_failure,
    open_attempt_stream,
    wrap_chat_stream,
)
from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    Admitted,
    CoordinationUnavailable,
    DynamicQueue,
    Grant,
    LocalQueueBackend,
    MatchingFailure,
    Outcome,
    PassThrough,
    PollReply,
    QueueSettings,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_router import (
    Admission,
    DynamicQueueCoordinationError,
    DynamicQueueFullError,
    DynamicQueueRouter,
    DynamicQueueSettings,
    DynamicQueueTimeoutError,
    RequestQueue,
    current_request_queue,
)
from litellm.text_completions.streaming_iterator import TextCompletionStreamWrapper
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _finished, _reap
from tests.unit.router_utils.test_dynamic_queue_router import _RecordsBackoffs

CAPACITY: Final = "deployment:d1"
MESSAGES: Final = [{"role": "user", "content": "hi"}]


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

    async def retry_stream(self, *args: object, **options: object) -> object:
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


class _OuterStreamCore(_RecoveringCore):
    """What the real chat and text stream wrappers ask of their core while they report a stream's end."""

    def outcome_callbacks(self, kwargs: object, sync: bool) -> dict[str, bool]:
        return {}

    async def report_stream_success(self, *args: object) -> None:
        return None

    async def report_stream_failure(self, *args: object) -> None:
        return None


def _outer_attempt(op: str, admission: Admission) -> StreamAttempt:
    return StreamAttempt(
        core=_OuterStreamCore(),  # pyright: ignore[reportArgumentType]  # serves only the stream-end reporting
        op=op,
        request={"stream": True},
        kwargs={"stream": True},
        group="group",
        order=None,
        depth=0,
        level=0,
        deployment_id="d1",
        attempted_retries=0,
        admission=admission,
    )


async def _chat_stream(attempt: StreamAttempt) -> object:
    response: Final = await litellm.acompletion(
        model="openai/gpt-x",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="one two three four five",
        stream=True,
    )
    return await open_attempt_stream(attempt, response)


async def _text_stream(attempt: StreamAttempt) -> object:
    response: Final = await litellm.atext_completion(
        model="openai/gpt-x", prompt="hi", mock_response="one two three four five", stream=True
    )
    return await open_attempt_stream(attempt, response)


def _idle_bound_s(router: DynamicQueueRouter) -> float:
    return router.queue_settings.lease_ms * 3 / 1000


async def _advance_idle_bound(timer: ManualTimer, router: DynamicQueueRouter) -> None:
    """Steps shorter than the lease renewal interval, so the probe's lease stays renewed as it would in real time."""
    step: Final = 2.5
    for _ in range(int(_idle_bound_s(router) / step)):
        await timer.parked(count=2)
        timer.advance(step)


async def _watched(attempt: StreamAttempt) -> None:
    watching: Final = attempt.watching
    assert watching is not None
    await _finished(watching)


async def test_the_real_chat_stream_read_to_its_end_counts_as_the_probes_success() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("acompletion", probe)
    stream: Final = await _chat_stream(attempt)

    read: Final = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # the returned stream
    await _settled(attempt)

    assert read
    assert await _busy(router) is False


async def test_the_real_chat_stream_closed_after_some_output_gives_its_turn_back_without_credit() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("acompletion", probe)
    stream: Final = await _chat_stream(attempt)

    _ = await stream.__anext__()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the returned stream
    await stream.aclose()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # see above
    await _settled(attempt)

    assert await _busy(router) is True
    assert await _next_probe_granted_at_once(timer, router) is True


async def test_the_real_chat_stream_nobody_reads_gives_its_turn_back_after_an_idle_bound() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("acompletion", probe)
    _ = await _chat_stream(attempt)

    await _advance_idle_bound(timer, router)
    await _watched(attempt)

    assert await _busy(router) is True
    assert await _next_probe_granted_at_once(timer, router) is True


async def test_the_real_text_stream_its_reader_abandoned_gives_its_turn_back_and_still_streams() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("atext_completion", probe)
    stream: Final = await _text_stream(attempt)
    await timer.parked(count=2)
    _ = await stream.__anext__()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the returned stream

    await _advance_idle_bound(timer, router)
    await _advance_idle_bound(timer, router)
    await _watched(attempt)
    released: Final = await _next_probe_granted_at_once(timer, router)
    rest: Final = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # see above

    assert released is True
    assert rest


async def test_the_real_text_stream_read_across_an_idle_bound_keeps_its_turn() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("atext_completion", probe)
    stream: Final = await _text_stream(attempt)

    await timer.parked(count=2)
    _ = await stream.__anext__()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the returned stream
    await _advance_idle_bound(timer, router)
    await timer.parked(count=2)
    held: Final = not await _next_probe_granted_at_once(timer, router)
    _ = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # see above
    await _settled(attempt)

    assert held is True
    assert await _busy(router) is False


class _ReportsOnTheTimer:
    """A local queue whose outcome reports take half a second of the test's time to land, like a slow Redis."""

    def __init__(self, timer: ManualTimer) -> None:
        self.timer = timer
        self.inner = LocalQueueBackend(QueueSettings(required_successes=1), clock=timer.now)
        self.reported = False

    @property
    def settings(self) -> QueueSettings:
        return self.inner.settings

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await self.inner.admit(request, budget_ms, ready_in_ms)

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        return await self.inner.poll(capacity, waiter_id, ready_in_ms)

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        return await self.inner.renew(capacity, waiter_id, token)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        await self.timer.sleep(0.5)
        await self.inner.report(capacity, waiter_id, grant, outcome)
        self.reported = True
        return None

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        return await self.inner.abandon(capacity, waiter_id)


class _RecordsWhetherTheFailureLanded(_RecoveringCore):
    def __init__(self, backend: _ReportsOnTheTimer) -> None:
        super().__init__()
        self.backend = backend
        self.landed_before_recovery: list[bool] = []

    async def recover_stream(self, *args: object) -> object:
        self.landed_before_recovery.append(self.backend.reported)
        return await super().recover_stream(*args)

    async def retry_stream(self, *args: object) -> object:
        self.landed_before_recovery.append(self.backend.reported)
        return await super().retry_stream(*args)


@pytest.mark.parametrize("recovery", ["fallback", "retry"])
async def test_a_stream_failure_reaches_the_queue_before_its_recovery_admits_anything(recovery: str) -> None:
    timer: Final = ManualTimer()
    backend: Final = _ReportsOnTheTimer(timer)
    await backend.inner.report(CAPACITY, "congests", PassThrough(0), MatchingFailure())
    timer.advance(1.0)
    queue: Final = DynamicQueue(backend, timer)
    request: Final = WaitRequest(CAPACITY, "probe", 0)
    admitted: Final = await queue.admit(request, 30.0)
    assert isinstance(admitted, Admitted)
    held: Final = contextlib.AsyncExitStack()
    router: Final = DynamicQueueRouter(DynamicQueueSettings(enabled=True), redis=lambda: None, timer=timer)
    admission: Final = Admission(await held.enter_async_context(queue.attempt(request, admitted)), held, router)
    core: Final = _RecordsWhetherTheFailureLanded(backend)
    attempt: Final = _stream_attempt(admission, router.request(0, "r"), core)

    _ = attempt.settle_admission(_provider_error(429))
    recovering: Final = asyncio.create_task(
        attempt.fallback(RuntimeError("cut")) if recovery == "fallback" else attempt.retry(RuntimeError("cut"))
    )
    try:
        await timer.parked(count=2)
        timer.advance(0.5)
        _ = await _finished(recovering)
    finally:
        await _reap(recovering)

    assert core.landed_before_recovery == [True]


async def test_a_queue_rejection_in_a_stream_recovery_ends_the_stream_with_that_rejection() -> None:
    rejection: Final = DynamicQueueFullError("m")

    class _RefusingCore(_RecoveringCore):
        async def recover_stream(self, *args: object) -> object:
            raise rejection

    attempt: Final = _stream_attempt(None, None, _RefusingCore())

    with pytest.raises(DynamicQueueFullError) as raised:
        _raise_fallback_failure(await _recovered_or_error(attempt))

    assert raised.value is rejection


async def _recovered_or_error(attempt: StreamAttempt) -> Exception:
    try:
        _ = await attempt.fallback(RuntimeError("cut"))
    except Exception as error:  # noqa: BLE001  # the error the wrapper hands to its fallback failure
        return error
    raise AssertionError("the recovery was refused")


def _chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        model="gpt-x", choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content=text))]
    )


class _SlowUpstream:
    """A provider stream whose second chunk arrives only when the test releases it."""

    def __init__(self) -> None:
        self.asked_for_more = asyncio.Event()
        self.release = asyncio.Event()

    async def chunks(self, first_too: bool) -> AsyncIterator[object]:
        if first_too:
            yield _chunk("first")
        self.asked_for_more.set()
        await self.release.wait()
        yield _chunk("second")


async def _chat_wrapper_over(attempt: StreamAttempt, rest: AsyncIterator[object]) -> AsyncIterator[object]:
    model_response: Final = await litellm.acompletion(
        model="openai/gpt-x", messages=[{"role": "user", "content": "hi"}], mock_response="unused", stream=True
    )
    return wrap_chat_stream(attempt, model_response, _FirstChunk(chunk=_chunk("first"), rest=rest))  # pyright: ignore[reportArgumentType]  # the mock call answers a stream


async def test_a_chat_stream_waiting_on_a_slow_provider_read_keeps_its_turn_past_the_idle_bound() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("acompletion", probe)
    upstream: Final = _SlowUpstream()
    stream: Final = await _chat_wrapper_over(attempt, upstream.chunks(first_too=False))
    read: Final[list[object]] = []

    async def consume() -> None:
        async for chunk in stream:
            read.append(chunk)

    reader: Final = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(upstream.asked_for_more.wait(), 10)
        await _advance_idle_bound(timer, router)
        await _advance_idle_bound(timer, router)
        await timer.parked(count=2)
        held: Final = not await _next_probe_granted_at_once(timer, router)
        upstream.release.set()
        await _finished(reader)
    finally:
        await _reap(reader)
    await _settled(attempt)

    assert held is True
    assert len(read) == 2
    assert await _busy(router) is False


async def test_a_text_stream_paused_between_chunks_gives_its_turn_back_and_keeps_streaming() -> None:
    timer: Final = ManualTimer()
    router, probe = await _probe(timer)
    attempt: Final = _outer_attempt("atext_completion", probe)
    upstream: Final = _SlowUpstream()
    stream: Final = await open_attempt_stream(
        attempt, TextCompletionStreamWrapper(completion_stream=upstream.chunks(first_too=True), model="gpt-x")
    )
    await timer.parked(count=2)
    _ = await stream.__anext__()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the returned stream

    await _advance_idle_bound(timer, router)
    await _advance_idle_bound(timer, router)
    await _watched(attempt)
    released: Final = await _next_probe_granted_at_once(timer, router)
    upstream.release.set()
    rest: Final = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # see above

    assert upstream.asked_for_more.is_set()
    assert released is True
    assert len(rest) == 1


class _CannotPublish(_ReportsOnTheTimer):
    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return CoordinationUnavailable(reason="redis_unavailable")


@pytest.mark.parametrize(
    ("failure", "recovers"),
    [
        (_provider_error(429), False),
        (litellm.RateLimitError(message="deployment at capacity", llm_provider="", model="m"), True),
    ],
    ids=["congesting_failure", "failure_that_congests_nothing"],
)
async def test_a_stream_failure_the_queue_never_learned_of_ends_its_recovery_as_unavailable(
    failure: Exception, recovers: bool
) -> None:
    timer: Final = ManualTimer()
    backend: Final = _CannotPublish(timer)
    await backend.inner.report(CAPACITY, "congests", PassThrough(0), MatchingFailure())
    timer.advance(1.0)
    queue: Final = DynamicQueue(backend, timer)
    request: Final = WaitRequest(CAPACITY, "probe", 0)
    admitted: Final = await queue.admit(request, 30.0)
    assert isinstance(admitted, Admitted)
    held: Final = contextlib.AsyncExitStack()
    router: Final = DynamicQueueRouter(DynamicQueueSettings(enabled=True), redis=lambda: None, timer=timer)
    admission: Final = Admission(await held.enter_async_context(queue.attempt(request, admitted)), held, router)
    core: Final = _RecoveringCore()
    attempt: Final = _stream_attempt(admission, router.request(0, "r"), core)

    _ = attempt.settle_admission(failure)
    if recovers:
        assert await attempt.fallback(RuntimeError("cut")) == "recovered"
    else:
        with pytest.raises(DynamicQueueCoordinationError):
            _ = await attempt.fallback(RuntimeError("cut"))

    assert len(core.recovered_with) == (1 if recovers else 0)


async def test_the_real_chat_stream_failing_before_content_ends_with_its_recoverys_queue_rejection() -> None:
    rejection: Final = DynamicQueueFullError("gpt-x")

    class _RefusingOuterCore(_OuterStreamCore):
        async def recover_stream(self, *args: object) -> object:
            raise rejection

    async def cut_before_content() -> AsyncIterator[object]:
        raise MidStreamFallbackError(
            message="upstream cut",
            model="gpt-x",
            llm_provider="openai",
            original_exception=_provider_error(500),
            generated_content="",
            is_pre_first_chunk=False,
        )
        yield _chunk("never")

    attempt: Final = StreamAttempt(
        core=_RefusingOuterCore(),  # pyright: ignore[reportArgumentType]  # serves only the stream's recovery
        op="acompletion",
        request={"stream": True},
        kwargs={"stream": True},
        group="group",
        order=None,
        depth=0,
        level=0,
        deployment_id="d1",
        attempted_retries=0,
    )
    model_response: Final = await litellm.acompletion(
        model="openai/gpt-x", messages=[{"role": "user", "content": "hi"}], mock_response="unused", stream=True
    )
    role_only: Final = ModelResponseStream(
        model="gpt-x", choices=[StreamingChoices(index=0, delta=Delta(role="assistant"))]
    )
    stream: Final = wrap_chat_stream(attempt, model_response, _FirstChunk(chunk=role_only, rest=cut_before_content()))  # pyright: ignore[reportArgumentType]  # the mock call answers a stream

    with pytest.raises(DynamicQueueFullError) as raised:
        _ = [chunk async for chunk in stream]

    assert raised.value is rejection


@pytest.mark.parametrize(("status", "backoffs"), [(429, 0), (503, 1)], ids=["congesting", "not_a_trigger"])
async def test_a_stream_retry_after_a_failure_the_queue_took_waits_there_instead_of_the_retry_backoff(
    status: int, backoffs: int
) -> None:
    host: Final = _RecordsBackoffs()
    arguments: Final = router_arguments(
        model_list=[{"model_name": "group", "litellm_params": {"model": "openai/gpt-x"}, "model_info": {"id": "d1"}}],
        default_priority=0,
        num_retries=1,
    )
    router: Final = litellm.Router.from_config(build_router_config(arguments), host)
    queue: Final = DynamicQueueRouter(
        DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=0.01), redis=lambda: None
    )
    request_queue: Final = queue.request(priority=0, request_id="stream")
    admission: Final = await request_queue.admit(CAPACITY, "group", _no_cooldown)
    attempt: Final = StreamAttempt(
        core=router._core,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the router's own Rust core
        op="acompletion",
        request={"model": "group", "messages": MESSAGES, "stream": True, "mock_response": "ok"},
        kwargs={"stream": True},
        group="group",
        order=None,
        depth=0,
        level=0,
        deployment_id="d1",
        attempted_retries=0,
        admission=admission,
        request_queue=request_queue,
    )
    failure: Final = _provider_error(status)
    _ = attempt.settle_admission(failure)

    result: Final = await attempt.retry(failure)
    if isinstance(result, tuple):
        stream: Final = result[0]
        assert isinstance(stream, litellm.CustomStreamWrapper)
        try:
            _ = [chunk async for chunk in stream]
        finally:
            await stream.aclose()

    assert isinstance(result, DynamicQueueTimeoutError if status == 429 else tuple)
    assert len(host.backoffs) == backoffs
