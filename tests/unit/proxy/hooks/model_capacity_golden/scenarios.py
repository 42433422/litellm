from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Protocol, runtime_checkable

import pytest

import litellm
from litellm import DualCache, Router
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.rate_limits import RateLimitScope, RateLimitStore
from litellm.types.utils import PriorityReservationSettings
from tests.unit.proxy.hooks.native_capacity_handler import fake_redis, listening


class Limiter(Protocol):
    @property
    def rate_limits(self) -> RateLimitStore: ...


@runtime_checkable
class CapacityHandler(Protocol):
    @property
    def v3_limiter(self) -> Limiter: ...

    def update_variables(self, llm_router: Router) -> None: ...

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None: ...


PreCall = Callable[[CapacityHandler, UserAPIKeyAuth, dict[str, object]], Awaitable[object]]
Build = Callable[[DualCache], CapacityHandler]
Scenario = Callable[[pytest.MonkeyPatch, Build, PreCall], Awaitable[object]]

_MODEL: Final = "shared-model"


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": _MODEL,
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "k", "api_base": "b", "tpm": 1000, "rpm": 100},
            }
        ]
    )


def _borrowing(monkeypatch: pytest.MonkeyPatch, fail_closed: bool) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"interactive": 0.6, "batch": 0.4})
    monkeypatch.setattr(
        litellm,
        "priority_reservation_settings",
        PriorityReservationSettings(
            saturation_threshold=0.1, team_model_limit_borrows_idle_capacity_from_other_teams=True
        ),
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", {"fail_closed_rate_limit_enforcement": fail_closed}
    )


def _team_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-borrow",
        team_id="team-a",
        metadata={"priority": "interactive"},
        team_metadata={"model_tpm_limit": {_MODEL: 100}},
    )


def _request() -> dict[str, object]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


def _outcome(error: BaseException | None) -> object:
    if error is None:
        return None
    cause: Final = error.__cause__
    return (type(error).__name__, getattr(error, "status_code", None), None if cause is None else type(cause).__name__)


def _unreachable_port() -> int:
    with listening() as (_, port):
        return port


async def _attempt(call: Awaitable[object]) -> object:
    try:
        await call
    except Exception as error:
        return _outcome(error)
    return None


async def batch_trips(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=False)
    seen: Final[list[object]] = []  # mutable-ok: each request's round trips and the scripts it sent
    with fake_redis(emulate=True) as redis:
        handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=redis.port)))
        handler.update_variables(llm_router=_router())
        for _ in range(3):
            trips: int = redis.trips
            sent: int = len(redis.commands)
            with request_redis_batch_scope():
                outcome = await _attempt(pre_call(handler, _team_key(), _request()))
            scripts = [
                tuple(part.decode() for part in command[3 : 3 + int(command[2])])
                for command in redis.commands[sent:]
                if command[0].upper() in (b"EVAL", b"EVALSHA")
            ]
            seen.append((outcome, redis.trips - trips, scripts))
    return seen[1:]


async def redis_down_fails_open(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=False)
    handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=_unreachable_port())))
    handler.update_variables(llm_router=_router())
    return [await _attempt(pre_call(handler, _team_key(), _request())) for _ in range(2)]


async def redis_down_fails_closed(monkeypatch: pytest.MonkeyPatch, build: Build, pre_call: PreCall) -> object:
    _borrowing(monkeypatch, fail_closed=True)
    handler: Final = build(DualCache(redis_cache=RedisCache(host="127.0.0.1", port=_unreachable_port())))
    handler.update_variables(llm_router=_router())
    admitted: Final = await _attempt(pre_call(handler, _team_key(), _request()))
    handler.v3_limiter.rate_limits.set_local(
        RateLimitScope("model_saturation_check", _MODEL).counter("tokens"), 150, 60
    )
    framed: Final = await _attempt(
        handler.enforce_token_limits_for_frame(user_api_key_dict=_team_key(), requested_model=_MODEL)
    )
    return [admitted, framed]


SCENARIOS: Final[Mapping[str, Scenario]] = {
    "test_a_request_batch_takes_the_round_trips_python_takes": batch_trips,
    "test_a_redis_outage_fails_open_like_python": redis_down_fails_open,
    "test_a_redis_outage_under_fail_closed_is_handled_like_python": redis_down_fails_closed,
}
