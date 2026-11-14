import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.total_deadline import total_deadline
from litellm.router_utils.dynamic_queue import (
    CoordinationUnavailable,
    MatchingFailure,
    OtherFailure,
    QueueFull,
    QueueSettings,
    QueueTimeout,
)
from litellm.router_utils.dynamic_queue_router import (
    QUEUE_COORDINATION_UNAVAILABLE_CODE,
    QUEUE_FULL_CODE,
    Admission,
    DynamicQueueCoordinationError,
    DynamicQueueRouter,
    DynamicQueueSettings,
    DynamicQueueTimeoutError,
    bound_request_queue,
    capacity_of,
    congested_ids,
    parse_dynamic_queue,
    prefer_uncongested,
    refuse_local_queue_across_workers,
    rejection_error,
)
from tests.unit.router_utils.test_dynamic_queue import ManualTimer, _reap

LOCAL: Final = DynamicQueueSettings(enabled=True, coordination="local")


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


def _router(timer: ManualTimer, settings: DynamicQueueSettings = LOCAL) -> DynamicQueueRouter:
    return DynamicQueueRouter(settings, redis=lambda: None, timer=timer)


async def _no_cooldown() -> float:
    return 0.0


def _cooldown(seconds: float) -> "Callable[[], Coroutine[object, object, float]]":
    async def left() -> float:
        return seconds

    return left


def _everywhere(deployment: Mapping[str, object]) -> bool:
    return True


async def _waited(
    timer: ManualTimer, call: Coroutine[object, object, Admission], step: float
) -> Admission | BaseException:
    """Drives an admission, moving time ``step`` at a time whenever it is parked on a recheck, and returns the
    admission or the error it raised."""
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
    await admission.finish(MatchingFailure(retry_after_ms=retry_after_ms))


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
        {"recovery_successes": 0},
        {"initial_backoff": 5, "max_backoff": 2},
    ],
    ids=["bool", "text", "success_status", "past_599", "no_successes", "cap_below_start"],
)
def test_an_invalid_recovery_or_trigger_setting_is_refused(block: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_dynamic_queue({"enabled": True, **block}, 0)


def test_the_recovery_settings_an_operator_sets_drive_the_queue() -> None:
    settings: Final = parse_dynamic_queue(
        {
            "enabled": True,
            "trigger_status_codes": [],
            "recovery_successes": 3,
            "initial_backoff": 0.5,
            "max_backoff": 8,
            "max_waiting": 40,
        },
        0,
    )

    assert settings is not None and settings.trigger_status_codes == ()
    assert settings.queue_settings() == QueueSettings(
        required_successes=3, initial_backoff_ms=500, max_backoff_ms=8_000, max_waiting=40
    )


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
    ("cooldown_left_s", "probe_runs_at"),
    [(0.0, 1.0), (4.0, 5.0)],
    ids=["deployment_without_cooldown", "deployment_still_cooling_down"],
)
async def test_a_probe_waits_out_what_is_left_of_its_deployments_own_cooldown(
    cooldown_left_s: float, probe_runs_at: float
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    await _congest(router, "cap", retry_after_ms=0)

    probe: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", _cooldown(cooldown_left_s)), 0.25)
    assert isinstance(probe, Admission)
    await probe.finish(None)

    assert timer.at == probe_runs_at


async def test_a_probe_whose_cooldown_outlasts_the_queue_time_left_times_out_and_hands_the_probe_on() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=3.0))
    await _congest(router, "cap", retry_after_ms=0)

    refused: Final = await _waited(timer, router.request(0, "r").admit("cap", "m", _cooldown(60.0)), 0.25)
    handed_on: Final = await _waited(timer, router.request(0, "next").admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(handed_on, Admission)
    await handed_on.finish(None)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 1.0


async def test_attempts_a_healthy_capacity_admits_cost_none_of_the_requests_queue_time() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=3.0))
    request_queue: Final = router.request(priority=0, request_id="r")
    for _ in range(3):
        healthy = await request_queue.admit("cap", "m", _no_cooldown)
        timer.advance(1.0)
        await healthy.finish(None)
    await _congest(router, "cap", retry_after_ms=60_000)

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 0.25)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 6.0


async def test_queue_time_is_spent_once_across_every_wait_of_one_request() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=3.0))
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
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)

    async with total_deadline(3_600, from_body=True, clock=timer.now):
        waiting: Final = router.request(priority=0, request_id="r").admit("cap", "m", _no_cooldown)
        refused: Final = await _waited(timer, waiting, 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_a_wait_after_the_requests_own_context_ended_still_ends_at_its_total_timeout() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer, DynamicQueueSettings(enabled=True, coordination="local", queue_timeout=7_200))
    await _congest(router, "cap", retry_after_ms=10_000_000)
    async with total_deadline(3_600, from_body=True, clock=timer.now):
        request_queue: Final = router.request(priority=0, request_id="r")

    refused: Final = await _waited(timer, request_queue.admit("cap", "m", _no_cooldown), 100.0)

    assert isinstance(refused, DynamicQueueTimeoutError)
    assert timer.at == 3_600


async def test_unknown_congestion_is_never_taken_for_a_healthy_capacity() -> None:
    router: Final = _router(ManualTimer(), DynamicQueueSettings(enabled=True, coordination="redis"))

    with bound_request_queue(router.request(priority=0, request_id="r")):
        busy: Final = await congested_ids(("d1",), lambda ids: (_deployment("d1"),), _everywhere)
        preferred: Final = await prefer_uncongested(("d1", "d2"), lambda ids: (_deployment("d1"),), _everywhere)

    assert busy == CoordinationUnavailable(reason="redis_not_configured")
    assert preferred == ("d1", "d2")


async def test_a_redis_queue_with_no_redis_refuses_the_wait_instead_of_queueing_in_one_process() -> None:
    router: Final = _router(ManualTimer(), DynamicQueueSettings(enabled=True, coordination="redis"))

    with pytest.raises(DynamicQueueCoordinationError) as refused:
        _ = await router.request(priority=0, request_id="r").admit("cap", "m", _no_cooldown)

    assert (refused.value.status_code, refused.value.code) == (503, QUEUE_COORDINATION_UNAVAILABLE_CODE)


def test_a_full_queue_answers_503_queue_full_and_running_out_of_time_is_the_usual_timeout() -> None:
    full: Final = rejection_error(QueueFull(), "m")
    late: Final = rejection_error(QueueTimeout(), "m")

    assert isinstance(full, litellm.ServiceUnavailableError)
    assert (full.status_code, getattr(full, "code", None)) == (503, QUEUE_FULL_CODE)
    assert isinstance(late, litellm.Timeout)
    assert late.status_code == 408


@pytest.mark.parametrize(
    ("coordination", "workers", "refused"),
    [("local", "2", True), ("local", "1", False), ("redis", "4", False)],
)
def test_a_proxy_with_several_workers_refuses_a_process_local_queue(
    coordination: str, workers: str, refused: bool
) -> None:
    settings: Final = {"enabled": True, "coordination": coordination}
    if refused:
        with pytest.raises(ValueError, match="local"):
            refuse_local_queue_across_workers(settings, workers)
    else:
        refuse_local_queue_across_workers(settings, workers)
    refuse_local_queue_across_workers({"enabled": False, "coordination": "local"}, "8")


def _deployment(deployment_id: str, capacity: str | None = None) -> Mapping[str, object]:
    shared: Final = {} if capacity is None else {"dynamic_queue_capacity_id": capacity}
    return {"model_name": "group", "model_info": {"id": deployment_id, **shared}}


def test_deployments_naming_one_capacity_share_it_apart_from_any_deployment_spelled_the_same() -> None:
    assert capacity_of(_deployment("d1", "shared-quota")) == capacity_of(_deployment("d2", "shared-quota"))
    assert capacity_of(_deployment("d3")) != capacity_of(_deployment("d4"))
    assert capacity_of(_deployment("d5", "pool")) != capacity_of(_deployment("pool"))


@pytest.mark.parametrize(
    ("congested", "kept"),
    [(("pool-a",), ("d3",)), (("pool-a", "pool-b"), ("d1", "d2", "d3")), ((), ("d1", "d2", "d3"))],
)
async def test_a_pick_prefers_deployments_whose_capacity_is_not_congested(
    congested: tuple[str, ...], kept: tuple[str, ...]
) -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    records: Final = {
        "d1": _deployment("d1", "pool-a"),
        "d2": _deployment("d2", "pool-a"),
        "d3": _deployment("d3", "pool-b"),
    }
    for pool in congested:
        await _congest(router, capacity_of(_deployment("any", pool)) or "", retry_after_ms=60_000)

    def lookup(ids: Sequence[str]) -> Sequence[Mapping[str, object]]:
        return tuple(records[deployment] for deployment in ids)

    with bound_request_queue(router.request(priority=0, request_id="r")):
        preferred: Final = await prefer_uncongested(("d1", "d2", "d3"), lookup, _everywhere)

    assert preferred == kept


async def test_a_deployment_the_request_reaches_on_its_callers_credentials_is_never_held_back() -> None:
    timer: Final = ManualTimer()
    router: Final = _router(timer)
    own_key: Final = _deployment("d1", "pool-a")
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


async def test_a_router_with_a_redis_queue_but_no_redis_refuses_the_request_with_its_503_code() -> None:
    router: Final = litellm.Router(
        model_list=_model_list(), default_priority=0, num_retries=2, dynamic_queue={"enabled": True}
    )

    with pytest.raises(DynamicQueueCoordinationError) as refused:
        _ = await router.acompletion(model="group", messages=[{"role": "user", "content": "hi"}])

    assert (refused.value.status_code, refused.value.code) == (503, QUEUE_COORDINATION_UNAVAILABLE_CODE)


@pytest.mark.parametrize("call", ["acompletion", "atext_completion"])
async def test_a_router_with_a_local_queue_serves_a_healthy_request_straight_through(call: str) -> None:
    router: Final = litellm.Router(
        model_list=_model_list(), default_priority=0, dynamic_queue={"enabled": True, "coordination": "local"}
    )
    request: Final = {"messages": [{"role": "user", "content": "hi"}]} if call == "acompletion" else {"prompt": "hi"}

    response: Final = await getattr(router, call)(model="group", **request)

    assert "served" in str(response)
