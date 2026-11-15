import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from typing import Final, TypeVar

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.total_deadline import total_deadline
from litellm.router import router_arguments
from litellm.router_config import build_router_config
from litellm.router_python_host import PythonHost
from litellm.router_rust.invoke.slots import max_parallel_requests_limit
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    CoordinationUnavailable,
    Grant,
    LocalQueueBackend,
    MatchingFailure,
    OtherFailure,
    Outcome,
    PollReply,
    QueueFull,
    QueueSettings,
    QueueTimeout,
    Succeeded,
    WaitRequest,
)
from litellm.router_utils.dynamic_queue_router import (
    QUEUE_FULL_CODE,
    Admission,
    Backends,
    DynamicQueueRouter,
    DynamicQueueSettings,
    DynamicQueueTimeoutError,
    bound_request_queue,
    capacity_of,
    congested_ids,
    on_redis,
    parse_dynamic_queue,
    prefer_uncongested,
    rejection_error,
)
from litellm.types.router import RouterRateLimitError
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _finished, _reap
from tests.unit.router_utils.test_dynamic_queue_redis import UnreachableRedis

ENABLED: Final = DynamicQueueSettings(enabled=True)
_T = TypeVar("_T")


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


def _in_memory(timer: ManualTimer, successes: int | None = None) -> Backends:
    """The queue coordinated in this process on the test's clock, standing in for its Redis."""

    def backend(settings: QueueSettings) -> LocalQueueBackend:
        tuned: Final = settings if successes is None else dataclasses.replace(settings, required_successes=successes)
        return LocalQueueBackend(tuned, clock=timer.now)

    return backend


def _router(
    timer: ManualTimer, settings: DynamicQueueSettings = ENABLED, successes: int | None = None
) -> DynamicQueueRouter:
    return DynamicQueueRouter(settings, _in_memory(timer, successes), timer=timer)


async def _no_cooldown() -> float:
    return 0.0


def _cooling_until(timer: ManualTimer, ends_at: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        return max(ends_at - timer.now(), 0.0)

    return left


def _everywhere(deployment: Mapping[str, object]) -> bool:
    return True


async def _waited(timer: ManualTimer, call: Coroutine[object, object, _T], step: float) -> _T | BaseException:
    """Drives a call, moving time ``step`` at a time whenever it is parked on a queue recheck, and returns what it
    returned or the error it raised."""
    task: Final = asyncio.create_task(call)
    try:
        for _ in range(10_000):
            await timer.parked(task)
            if task.done():
                return task.exception() or task.result()
            timer.advance(step)
        raise AssertionError("deadlock safeguard: the admission never settled")
    finally:
        await _reap(task)


async def _congest(router: DynamicQueueRouter, capacity: str, retry_after_ms: int) -> None:
    request_queue: Final = router.request(priority=0, request_id=f"congest-{capacity}")
    admission: Final = await request_queue.admit(capacity, "m", _no_cooldown)
    assert admission is not None
    assert await admission.finish(MatchingFailure(retry_after_ms=retry_after_ms)) is True


def test_enabling_the_queue_needs_a_default_priority() -> None:
    with pytest.raises(ValueError, match="default_priority"):
        parse_dynamic_queue({"enabled": True}, None)

    assert parse_dynamic_queue({"enabled": True}, -3) == DynamicQueueSettings(enabled=True)
    assert parse_dynamic_queue({"enabled": False}, None) is None
    assert parse_dynamic_queue(None, 0) is None


def test_an_unknown_dynamic_queue_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_dynamic_queue({"enabled": True, "queue_timeot": 5}, 0)


@pytest.mark.parametrize(
    "block",
    [
        {"trigger_status_codes": [True]},
        {"trigger_status_codes": ["429"]},
        {"trigger_status_codes": [200]},
        {"trigger_status_codes": [600]},
        {"queue_timeout": float("inf")},
        {"queue_timeout": float("nan")},
        {"max_waiting": 0},
        {"coordination": "local"},
        {"recovery_successes": 3},
        {"initial_backoff": 0.5},
    ],
    ids=[
        "bool",
        "text",
        "success_status",
        "past_599",
        "endless_queue",
        "nan_queue",
        "no_room",
        "coordination_is_not_a_setting",
        "recovery_is_not_a_setting",
        "backoff_is_not_a_setting",
    ],
)
def test_an_invalid_or_internal_dynamic_queue_setting_is_refused(block: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_dynamic_queue({"enabled": True, **block}, 0)


def test_the_queue_limits_an_operator_sets_drive_the_queue_with_internal_recovery_defaults() -> None:
    settings: Final = parse_dynamic_queue(
        {"enabled": True, "trigger_status_codes": [], "queue_timeout": 12, "max_waiting": 40}, 0
    )

    assert settings is not None and settings.trigger_status_codes == ()
    assert settings.queue_timeout == 12
    assert settings.queue_settings() == dataclasses.replace(QueueSettings(), max_waiting=40)


@pytest.mark.parametrize(
    ("triggers", "status_code", "congests"),
    [
        ((429, 529), 429, True),
        ((429, 529), 529, True),
        ((429, 529), 503, False),
        ((), 429, False),
        ((503,), 503, True),
    ],
)
def test_only_a_provider_answering_with_a_trigger_status_congests(
    triggers: tuple[int, ...], status_code: int, congests: bool
) -> None:
    router: Final = _router(ManualTimer(), DynamicQueueSettings(enabled=True, trigger_status_codes=triggers))

    outcome: Final = router.outcome_of(_provider_error(status_code), None)

    assert isinstance(outcome, MatchingFailure) is congests
    assert isinstance(outcome, OtherFailure) is not congests


def test_a_rate_limit_litellm_raised_itself_never_congests() -> None:
    synthetic: Final = litellm.RateLimitError(message="deployment at capacity", llm_provider="", model="m")

    assert isinstance(_router(ManualTimer()).outcome_of(synthetic, 30.0), OtherFailure)


@pytest.mark.parametrize(("retry_after_s", "waits_ms"), [(None, 0), (12.0, 12_000)])
def test_a_congesting_failure_holds_the_next_probe_for_its_providers_retry_after(
    retry_after_s: float | None, waits_ms: int
) -> None:
    assert _router(ManualTimer()).outcome_of(_provider_error(429), retry_after_s) == MatchingFailure(
        retry_after_ms=waits_ms
    )


@pytest.mark.parametrize(
    ("cooldown_ends_at", "probe_runs_at"),
    [(0.0, 1.0), (5.0, 5.0)],
    ids=["deployment_without_cooldown", "deployment_still_cooling_down"],
)
async def test_a_wait_stays_queued_until_its_deployments_own_cooldown_ends(
    cooldown_ends_at: float, probe_runs_at: float
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    await _congest(router, "cap", retry_after_ms=0)
    cooling: Final = _cooling_until(timer, cooldown_ends_at)

    probe: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", cooling), 0.25)
    assert isinstance(probe, Admission)
    await probe.finish(None)

    assert timer.at == probe_runs_at


async def test_a_wait_whose_cooldown_outlasts_the_queue_time_times_out_without_holding_the_probe() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=0)
    cooling: Final = _cooling_until(timer, 60.0)

    refused: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", cooling), 0.25)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3.0


async def test_attempts_a_healthy_capacity_admits_cost_none_of_the_requests_queue_time() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=3.0))
    request_queue: Final = router.request(priority=0, request_id="r")
    for _ in range(3):
        healthy = await request_queue.admit("cap", "m", _no_cooldown)
        assert healthy is not None
        timer.advance(1.0)
        _ = await healthy.finish(None)
    await _congest(router, "cap", retry_after_ms=60_000)

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 6.0


async def test_a_wait_through_a_recovery_drain_is_spent_from_the_requests_queue_time() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=3.0), successes=1)
    await _congest(router, "cap", retry_after_ms=1_000)
    timer.advance(1.0)
    probe: Final = await router.request(0, "probe").admit("cap", "m", _no_cooldown)
    assert probe is not None
    held: Final = asyncio.create_task(router.request(0, "held").admit("cap", "m", _no_cooldown))
    try:
        await timer.parked(held)
        await probe.finish(Succeeded())
        request_queue: Final = router.request(priority=0, request_id="r")
        drained: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)
        assert isinstance(drained, Admission)
        await drained.finish(None)
        released: Final = await _finished(held)
        assert released is not None
        _ = await released.finish(None)
    finally:
        await _reap(held)
    await _congest(router, "cap", retry_after_ms=60_000)

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 4.0


async def test_queue_time_is_spent_once_across_every_wait_of_one_request() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=1_000)
    request_queue: Final = router.request(priority=0, request_id="r")

    probe: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(probe, Admission)
    assert timer.at == 1.0
    await probe.finish(MatchingFailure(retry_after_ms=60_000))
    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3.0


async def test_a_wait_never_outlasts_the_requests_total_timeout() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)

    async with total_deadline(3_600, from_body=True, clock=timer.now):
        waiting: Final = router.request(priority=0, request_id="r").admit("cap", "m", _no_cooldown)
        refused: Final = await _waited(timer, waiting, 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_a_wait_after_the_requests_own_context_ended_still_ends_at_its_total_timeout() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)
    async with total_deadline(3_600, from_body=True, clock=timer.now):
        request_queue: Final = router.request(priority=0, request_id="r")

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_congestion_the_queue_cannot_read_leaves_every_candidate_and_the_request_off_the_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(ENABLED, on_redis(lambda: UnreachableRedis()), timer=timer)
    request_queue: Final = router.request(priority=0, request_id="r")

    with bound_request_queue(request_queue):
        preferred: Final = await prefer_uncongested(("d1", "d2"), lambda ids: (_deployment("d1"),), _everywhere)
        busy: Final = await congested_ids(("d1",), lambda ids: (_deployment("d1"),), _everywhere)
    admitted: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)

    assert preferred == ("d1", "d2")
    assert busy == frozenset()
    assert admitted is None


async def test_a_request_with_no_queue_backend_goes_on_without_the_queue_and_a_later_one_queues_again() -> None:
    timer: Final = ManualTimer()
    backends: Final[list[LocalQueueBackend]] = []  # mutable-ok: the Redis the proxy attaches after the first request
    router: Final = DynamicQueueRouter(ENABLED, lambda settings: backends[0] if backends else None, timer=timer)
    first: Final = router.request(priority=0, request_id="first")

    without_queue: Final = await first.admit("deployment:d1", "m", _no_cooldown)
    backends.append(LocalQueueBackend(router.queue_settings, clock=timer.now))
    still_without: Final = await first.admit("deployment:d1", "m", _no_cooldown)
    queued: Final = await router.request(priority=0, request_id="later").admit("deployment:d1", "m", _no_cooldown)

    assert without_queue is None
    assert still_without is None
    assert isinstance(queued, Admission)
    assert await queued.finish(Succeeded()) is True


class _ReportsUnreachable(LocalQueueBackend):
    """Admits in this process, then loses every report, like a Redis that drops after the turn was granted."""

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        return CoordinationUnavailable(reason="redis_unavailable")


async def test_a_report_the_queue_cannot_take_takes_the_rest_of_the_request_off_the_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = DynamicQueueRouter(
        ENABLED, lambda settings: _ReportsUnreachable(settings, clock=timer.now), timer=timer
    )
    request_queue: Final = router.request(priority=0, request_id="r")
    admission: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)
    assert admission is not None

    learned: Final = await admission.finish(MatchingFailure())
    retried: Final = await request_queue.admit("deployment:d1", "m", _no_cooldown)

    assert learned is False
    assert request_queue.coordinated is False
    assert retried is None


def test_a_full_queue_answers_503_queue_full_and_running_out_of_time_is_the_usual_timeout() -> None:
    full: Final = rejection_error(QueueFull(), "m")
    late: Final = rejection_error(QueueTimeout(), "m")

    assert isinstance(full, litellm.ServiceUnavailableError)
    assert (full.status_code, getattr(full, "code", None)) == (503, QUEUE_FULL_CODE)
    assert isinstance(late, litellm.Timeout)
    assert late.status_code == 408


def _deployment(deployment_id: str, **model_info: object) -> Mapping[str, object]:
    return {"model_name": "group", "model_info": {"id": deployment_id, **model_info}}


async def test_deployments_sharing_every_label_but_their_id_never_share_a_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    labels: Final = {"dynamic_queue_capacity_id": "one-account", "base_model": "same", "team": "same"}
    records: Final = {"d1": _deployment("d1", **labels), "d2": _deployment("d2", **labels)}
    await _congest(router, capacity_of(records["d1"]) or "", retry_after_ms=60_000)

    def lookup(ids: Sequence[str]) -> Sequence[Mapping[str, object]]:
        return tuple(records[deployment] for deployment in ids)

    with bound_request_queue(router.request(priority=0, request_id="r")):
        preferred: Final = await prefer_uncongested(("d1", "d2"), lookup, _everywhere)

    assert preferred == ("d2",)
    assert capacity_of(records["d1"]) != capacity_of(records["d2"])


@pytest.mark.parametrize(
    ("congested", "kept"),
    [(("d1", "d2"), ("d3",)), (("d1", "d2", "d3"), ("d1", "d2", "d3")), ((), ("d1", "d2", "d3"))],
)
async def test_a_pick_prefers_deployments_whose_queue_is_not_congested(
    congested: tuple[str, ...], kept: tuple[str, ...]
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    records: Final = {name: _deployment(name) for name in ("d1", "d2", "d3")}
    for name in congested:
        await _congest(router, capacity_of(records[name]) or "", retry_after_ms=60_000)

    def lookup(ids: Sequence[str]) -> Sequence[Mapping[str, object]]:
        return tuple(records[deployment] for deployment in ids)

    with bound_request_queue(router.request(priority=0, request_id="r")):
        preferred: Final = await prefer_uncongested(("d1", "d2", "d3"), lookup, _everywhere)

    assert preferred == kept


async def test_a_deployment_the_request_reaches_on_its_callers_credentials_is_never_held_back() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    own_key: Final = _deployment("d1")
    await _congest(router, capacity_of(own_key) or "", retry_after_ms=60_000)

    with bound_request_queue(router.request(priority=0, request_id="r")):
        busy: Final = await congested_ids(("d1",), lambda ids: (own_key,), lambda deployment: False)

    assert busy == frozenset()


async def test_a_request_that_does_not_queue_picks_as_today_without_reading_any_deployment() -> None:
    def lookup(ids: Sequence[str]) -> Sequence[Mapping[str, object]]:
        raise AssertionError(f"read {ids} for a request that does not queue")

    assert await prefer_uncongested(("d1", "d2"), lookup, _everywhere) == ("d1", "d2")


def _model_list() -> list[dict[str, object]]:
    return [{"model_name": "group", "litellm_params": {"model": "openai/gpt-x", "mock_response": "served"}}]


def test_a_router_enabling_the_queue_without_a_default_priority_fails_to_build() -> None:
    with pytest.raises(ValueError, match="default_priority"):
        litellm.Router(model_list=_model_list(), dynamic_queue={"enabled": True})


@pytest.mark.parametrize("call", ["acompletion", "atext_completion"])
async def test_a_router_whose_queue_has_no_redis_serves_a_request_as_it_would_without_the_queue(call: str) -> None:
    router: Final = litellm.Router(
        model_list=_model_list(), default_priority=0, num_retries=2, dynamic_queue={"enabled": True}
    )
    request: Final = {"messages": [{"role": "user", "content": "hi"}]} if call == "acompletion" else {"prompt": "hi"}

    response: Final = await getattr(router, call)(model="group", **request)

    assert "served" in str(response)


@pytest.mark.parametrize("call", ["acompletion", "atext_completion"])
async def test_a_router_with_a_queue_serves_a_healthy_request_straight_through(call: str) -> None:
    timer: Final = ManualTimer()
    router: Final = _on_the_clock(
        litellm.Router(model_list=_model_list(), default_priority=0, dynamic_queue={"enabled": True}), timer
    )
    request: Final = {"messages": [{"role": "user", "content": "hi"}]} if call == "acompletion" else {"prompt": "hi"}

    response: Final = await getattr(router, call)(model="group", **request)

    assert "served" in str(response)


def _ordered_pair() -> list[dict[str, object]]:
    return [
        {"model_name": "group", "litellm_params": {"model": "openai/gpt-x", "order": order}, "model_info": {"id": name}}
        for name, order in (("first", 1), ("second", 2))
    ]


def _on_the_clock(router: litellm.Router, timer: ManualTimer, backends: Backends | None = None) -> litellm.Router:
    """The router's own queue, built as its endpoints build it, on the test's clock and an in-process backend in
    place of its Redis."""
    endpoints: Final = router._core._endpoints  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the router's own endpoints
    settings: Final = endpoints._core.resolved.dynamic_queue  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # its resolved queue settings
    assert isinstance(settings, DynamicQueueSettings)
    endpoints._queue_router = DynamicQueueRouter(settings, backends or _in_memory(timer), timer=timer)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the queue it builds lazily
    return router


def _queueing_router(timer: ManualTimer, disable_cooldowns: bool = False) -> litellm.Router:
    router: Final = litellm.Router(
        model_list=_ordered_pair(),
        default_priority=0,
        num_retries=0,
        cooldown_time=60,
        disable_cooldowns=disable_cooldowns,
        dynamic_queue={"enabled": True, "queue_timeout": 0.5},
    )
    return _on_the_clock(router, timer)


MESSAGES: Final = [{"role": "user", "content": "hi"}]


async def test_a_group_whose_every_deployment_is_cooling_and_congested_waits_in_the_queue_instead_of_failing() -> None:
    timer: Final = ManualTimer()
    router: Final = _queueing_router(timer)
    with pytest.raises(litellm.RateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))

    waited: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, mock_response="never reached"), 0.25
    )

    assert isinstance(waited, DynamicQueueTimeoutError)
    assert timer.at == 0.5


async def test_a_pick_passes_over_a_congested_deployment_its_order_would_pick_for_a_healthy_one() -> None:
    timer: Final = ManualTimer()
    router: Final = _queueing_router(timer, disable_cooldowns=True)
    with pytest.raises(litellm.RateLimitError):
        _ = await router.acompletion(model="first", messages=MESSAGES, mock_response=_provider_error(429))

    served: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, mock_response="served"), 0.25
    )

    assert "served" in str(served)
    assert served._hidden_params.get("model_id") == "second"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the response's hidden params


async def test_a_probe_a_full_parallel_slot_refuses_gives_its_turn_back_without_ending_the_recovery() -> None:
    timer: Final = ManualTimer()
    queue: Final = _router(timer, successes=2)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "openai/gpt-x", "max_parallel_requests": 1},
                "model_info": {"id": "only"},
            }
        ],
        num_retries=0,
    )
    core: Final = router._core  # pyright: ignore[reportPrivateUsage]  # the router's own Rust core
    limit: Final = max_parallel_requests_limit(core._port, core._records_by_id(("only",))[0], {})  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]  # the slot the router keeps for the deployment
    assert isinstance(limit, MaxParallelRequestsLimit)

    async def call(request_id: str, answer: object) -> object:
        with bound_request_queue(queue.request(priority=0, request_id=request_id)):
            try:
                return await router.acompletion(model="group", messages=MESSAGES, mock_response=answer)
            except litellm.RateLimitError as refused:
                return refused

    congested: Final = await call("congests", _provider_error(429))
    timer.advance(1.0)
    first: Final = await call("first", "first success")
    with limit:
        refused: Final = await call("refused", "never reached")
    second: Final = await call("second", "second success")
    busy: Final = await queue.busy({"only": "deployment:only"})

    assert isinstance(congested, litellm.RateLimitError) and isinstance(refused, litellm.RateLimitError)
    assert "first success" in str(first) and "second success" in str(second)
    assert busy == frozenset()


class _RecordsBackoffs(PythonHost):
    def __init__(self) -> None:
        self.backoffs: list[float] = []  # mutable-ok: each retry backoff the router asked for

    def sleep(self, seconds: float) -> Awaitable[None]:
        self.backoffs.append(seconds)
        return _no_wait()


async def _no_wait() -> None:
    return None


def _retrying_router(
    host: PythonHost, timer: ManualTimer, *, queued: bool = True, backends: Backends | None = None
) -> litellm.Router:
    arguments: Final = router_arguments(
        model_list=[{"model_name": "group", "litellm_params": {"model": "openai/gpt-x"}, "model_info": {"id": "only"}}],
        default_priority=0,
        num_retries=1,
        dynamic_queue={"enabled": queued, "queue_timeout": 0.5},
    )
    router: Final = litellm.Router.from_config(build_router_config(arguments), host)
    return _on_the_clock(router, timer, backends) if queued else router


async def test_a_retry_after_a_congesting_failure_waits_in_the_queue_instead_of_the_retry_backoff() -> None:
    host: Final = _RecordsBackoffs()
    timer: Final = ManualTimer()
    router: Final = _retrying_router(host, timer)

    waited: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429)), 0.25
    )

    assert isinstance(waited, DynamicQueueTimeoutError)
    assert timer.at == 0.5
    assert host.backoffs == []


async def test_a_retry_after_a_failure_the_queue_does_not_handle_keeps_its_retry_backoff() -> None:
    queued: Final = _RecordsBackoffs()
    unqueued: Final = _RecordsBackoffs()
    for host, enabled in ((queued, True), (unqueued, False)):
        with pytest.raises(litellm.ServiceUnavailableError):
            _ = await _retrying_router(host, ManualTimer(), queued=enabled).acompletion(
                model="group", messages=MESSAGES, mock_response=_provider_error(503)
            )

    assert queued.backoffs
    assert len(queued.backoffs) == len(unqueued.backoffs)


@pytest.mark.parametrize(
    "backends",
    [
        lambda timer: on_redis(lambda: UnreachableRedis()),
        lambda timer: lambda settings: _ReportsUnreachable(settings, clock=timer.now),
    ],
    ids=["redis_unreachable_from_the_start", "redis_lost_after_the_turn_was_granted"],
)
async def test_a_congesting_failure_without_the_queues_redis_keeps_the_providers_error_and_its_retry_backoff(
    backends: Callable[[ManualTimer], Backends],
) -> None:
    timer: Final = ManualTimer()
    queued: Final = _RecordsBackoffs()
    unqueued: Final = _RecordsBackoffs()
    for host, router in (
        (queued, _retrying_router(queued, timer, backends=backends(timer))),
        (unqueued, _retrying_router(unqueued, timer, queued=False)),
    ):
        with pytest.raises(litellm.RateLimitError) as raised:
            _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))
        assert raised.value.status_code == 429

    assert queued.backoffs
    assert len(queued.backoffs) == len(unqueued.backoffs)


class _LosesRedis(LocalQueueBackend):
    """Coordinates in this process until the test cuts it off, then answers every operation as an unreachable Redis."""

    def __init__(self, settings: QueueSettings, timer: ManualTimer) -> None:
        super().__init__(settings, clock=timer.now)
        self.reachable = True

    async def admit(self, request: WaitRequest, budget_ms: int, ready_in_ms: int = 0) -> AdmitReply:
        return await super().admit(request, budget_ms, ready_in_ms) if self.reachable else _UNREACHABLE

    async def poll(self, capacity: str, waiter_id: str, ready_in_ms: int = 0) -> PollReply:
        return await super().poll(capacity, waiter_id, ready_in_ms) if self.reachable else _UNREACHABLE

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        return await super().abandon(capacity, waiter_id) if self.reachable else _UNREACHABLE

    async def busy(self, capacity: str) -> bool | CoordinationUnavailable:
        return await super().busy(capacity) if self.reachable else _UNREACHABLE


_UNREACHABLE: Final = CoordinationUnavailable(reason="redis_unavailable")


async def test_a_cooling_deployment_kept_for_the_queue_is_never_called_once_the_queues_redis_is_lost() -> None:
    timer: Final = ManualTimer()
    backends: Final[list[_LosesRedis]] = []  # mutable-ok: the one backend the router builds, kept to cut it off

    def backend(settings: QueueSettings) -> _LosesRedis:
        backends.append(_LosesRedis(settings, timer))
        return backends[-1]

    router: Final = _on_the_clock(
        litellm.Router(
            model_list=_ordered_pair(),
            default_priority=0,
            num_retries=0,
            cooldown_time=3_600,
            dynamic_queue={"enabled": True, "queue_timeout": 30},
        ),
        timer,
        backend,
    )
    with pytest.raises(litellm.RateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))
    waiting: Final = asyncio.create_task(
        router.acompletion(model="group", messages=MESSAGES, mock_response="never reached")
    )
    try:
        await timer.parked(waiting)
        backends[0].reachable = False
        timer.advance(0.25)
        outcome: Final = await asyncio.wait_for(asyncio.gather(waiting, return_exceptions=True), 10)
    finally:
        await _reap(waiting)

    assert isinstance(outcome[0], RouterRateLimitError)


@pytest.mark.parametrize(
    ("override", "served"),
    [({"api_key": "caller-own-key"}, True), ({"api_key": None}, False)],
    ids=["caller_key_on_the_fallback", "empty_api_key_field_on_the_fallback"],
)
async def test_a_fallback_target_sending_its_own_credentials_skips_the_queue_and_an_empty_field_waits_in_it(
    override: Mapping[str, object], served: bool
) -> None:
    timer: Final = ManualTimer()
    router: Final = _on_the_clock(
        litellm.Router(
            model_list=[
                {
                    "model_name": "primary",
                    "litellm_params": {"model": "openai/gpt-x", "mock_response": _provider_error(429)},
                    "model_info": {"id": "primary-1"},
                },
                {
                    "model_name": "backup",
                    "litellm_params": {"model": "openai/gpt-y", "mock_response": "served"},
                    "model_info": {"id": "backup-1"},
                },
            ],
            default_priority=0,
            num_retries=0,
            dynamic_queue={"enabled": True, "queue_timeout": 0.5},
        ),
        timer,
    )
    queue: Final = router._core._endpoints._dynamic_queue()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the queue the router waits in
    await _congest(queue, "deployment:backup-1", retry_after_ms=60_000)  # pyright: ignore[reportUnknownArgumentType]  # see above

    outcome: Final = await _waited(
        timer, router.acompletion(model="primary", messages=MESSAGES, fallbacks=[{"model": "backup", **override}]), 0.25
    )

    if served:
        assert "served" in str(outcome)
    else:
        assert isinstance(outcome, DynamicQueueTimeoutError)


async def test_a_group_left_only_cooling_with_the_queues_redis_unreachable_fails_as_it_would_without_the_queue() -> (
    None
):
    timer: Final = ManualTimer()
    router: Final = _on_the_clock(
        litellm.Router(
            model_list=_ordered_pair(),
            default_priority=0,
            num_retries=0,
            cooldown_time=60,
            dynamic_queue={"enabled": True, "queue_timeout": 0.5},
        ),
        timer,
        on_redis(lambda: UnreachableRedis()),
    )
    with pytest.raises(litellm.RateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response=_provider_error(429))

    with pytest.raises(RouterRateLimitError):
        _ = await router.acompletion(model="group", messages=MESSAGES, mock_response="never reached")


def _never_answers(started: asyncio.Event) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return left


async def test_a_cooldown_lookup_that_never_answers_lets_the_attempt_go_on_without_the_queue_after_its_timeout() -> (
    None
):
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    await _congest(router, "cap", retry_after_ms=0)
    timer.advance(1.0)
    looking: Final = asyncio.Event()
    request_queue: Final = router.request(0, "r")
    probe: Final = asyncio.create_task(request_queue.admit("cap", "m", _never_answers(looking)))
    try:
        await asyncio.wait_for(looking.wait(), 10)
        await timer.parked(count=1)
        timer.advance(router.queue_settings.operation_timeout_ms / 1000)
        unqueued: Final = await _finished(probe)
    finally:
        await _reap(probe)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert unqueued is None
    assert request_queue.coordinated is False


def _slow_cooldown_lookup(timer: ManualTimer, seconds: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        await timer.sleep(seconds)
        return 0.0

    return left


async def test_a_cooldown_lookup_slower_than_the_queue_time_left_times_out_and_dispatches_nothing() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=0.5))
    await _congest(router, "cap", retry_after_ms=0)
    timer.advance(1.0)
    probe: Final = asyncio.create_task(router.request(0, "r").admit("cap", "m", _slow_cooldown_lookup(timer, 0.75)))
    try:
        await timer.parked(count=2)
        timer.advance(0.5)
        refused: Final = await asyncio.wait_for(asyncio.gather(probe, return_exceptions=True), 10)
    finally:
        await _reap(probe)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)
    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(refused[0], DynamicQueueTimeoutError)
    assert timer.at == 1.5


@pytest.mark.parametrize(
    ("model", "named"),
    [("pinned", {}), ("anthropic/claude-sonnet-5-5", {"specific_deployment": True})],
    ids=["deployment_id", "specific_deployment"],
)
async def test_a_request_naming_one_deployment_on_forwarded_credentials_keeps_the_priority_scheduler(
    model: str, named: Mapping[str, object]
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "mock_response": "served"},
                "model_info": {"id": "pinned"},
            }
        ],
        default_priority=0,
        dynamic_queue={"enabled": True},
    )
    forwarded: Final = {"custom_llm_provider": "anthropic", "extra_headers": {"Authorization": "Bearer caller-own"}}

    response: Final = await router.acompletion(
        model=model, messages=MESSAGES, priority=1, provider_specific_header=forwarded, **named
    )

    headers: Final = response._hidden_params.get("additional_headers") or {}  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the response's hidden params
    assert headers.get("x-litellm-request-prioritization-used") is True  # pyright: ignore[reportUnknownMemberType]  # see above


@pytest.mark.parametrize(
    ("credentials", "scheduled"),
    [({}, False), ({"api_key": None}, False), ({"api_key": "caller-own-key"}, True)],
    ids=["none_sent", "empty_api_key_field", "caller_api_key"],
)
async def test_a_request_on_its_callers_own_credentials_keeps_the_priority_scheduler(
    credentials: Mapping[str, object], scheduled: bool
) -> None:
    router: Final = litellm.Router(model_list=_model_list(), default_priority=0, dynamic_queue={"enabled": True})

    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hi"}], priority=1, **credentials
    )

    headers: Final = response._hidden_params.get("additional_headers") or {}  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the response's hidden params
    assert headers.get("x-litellm-request-prioritization-used") is (True if scheduled else None)  # pyright: ignore[reportUnknownMemberType]  # see above


async def test_a_total_timeout_released_for_a_stream_never_bounds_its_queue_wait() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=10_000_000)

    async with total_deadline(1.0, from_body=False, clock=timer.now) as deadline:
        request_queue: Final = router.request(priority=0, request_id="r")
        deadline.release()
        refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3.0


async def test_an_empty_api_key_field_still_waits_in_its_configured_deployments_queue() -> None:
    timer: Final = ManualTimer()
    router: Final = _on_the_clock(
        litellm.Router(
            model_list=[
                {"model_name": "group", "litellm_params": {"model": "openai/gpt-x"}, "model_info": {"id": "vertex"}}
            ],
            default_priority=0,
            num_retries=0,
            dynamic_queue={"enabled": True, "queue_timeout": 0.5},
        ),
        timer,
    )
    queue: Final = router._core._endpoints._dynamic_queue()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # the queue the router waits in
    await _congest(queue, "deployment:vertex", retry_after_ms=60_000)  # pyright: ignore[reportUnknownArgumentType]  # see above

    waited: Final = await _waited(
        timer, router.acompletion(model="group", messages=MESSAGES, api_key=None, mock_response="never reached"), 0.25
    )

    assert isinstance(waited, DynamicQueueTimeoutError)
