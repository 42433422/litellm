import socket
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm import DualCache, Router
from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_capacity_native import (
    NativeDynamicRateLimiter,
    build_native_dynamic_rate_limiter,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    BorrowingTeamModelLimits,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitedModel,
    borrowing_team_model_limits,
    get_request_stash,
)
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import HookRun, InternalUsageCache, hook_fields, raise_v2_abort
from litellm.proxy.v2_proxy_fields import (
    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
    DERIVED_POST_CALL_SUCCESS_FIELDS,
    DERIVED_RESPONSE_HEADERS_FIELDS,
    PRE_CALL_SOURCES,
)
from litellm.redis.stores.rate_limits import LocalValue, RateLimitCounter, RateLimitScope, RateLimitStore
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome
from litellm.types.router import ModelGroupInfo


@dataclass(frozen=True, slots=True)
class TeamShare:
    share: RateLimitDescriptor
    ceiling: RateLimitDescriptor | None


def _team_descriptor(
    limiter: PROXY_MaxParallelRequestsHandler_v3, key: str, value: str, limits: BorrowingTeamModelLimits
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key=key,
        value=value,
        rate_limit={
            "tokens_per_unit": limits.tpm,
            "requests_per_unit": limits.rpm,
            "window_size": limiter.window_size,
        },
    )


def borrowing_team_share(
    handler: "NativeDynamicRateLimitHandler", model: str, auth: UserAPIKeyAuth
) -> TeamShare | None:
    limiter: Final = handler.v3_limiter
    rate_limited: Final = limiter.rate_limited_model(model) or RateLimitedModel(requested=model, group=model)
    limits: Final = borrowing_team_model_limits(auth, rate_limited)
    if limits.tpm is None and limits.rpm is None:
        return None
    value: Final = f"{auth.team_id}:{rate_limited.group}"
    ceiling: Final = limits.borrow_ceiling()
    return TeamShare(
        share=_team_descriptor(limiter, "model_per_team_saturated", value, limits),
        ceiling=None
        if ceiling is None
        else _team_descriptor(limiter, "model_per_team_max_borrow_multiple", value, ceiling),
    )


def model_tracking_descriptor(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key="model_saturation_check",
        value=model,
        rate_limit={
            "requests_per_unit": model_group_info.rpm or None,
            "tokens_per_unit": model_group_info.tpm or None,
            "window_size": handler.v3_limiter.window_size,
        },
    )


async def model_saturation(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> float:
    rate_limits: Final = handler.rate_limits
    scope: Final = RateLimitScope("model_saturation_check", model)
    requests: Final = await rate_limits.get_from_redis(scope.counter("requests"))
    tokens: Final = await rate_limits.get_from_redis(scope.counter("tokens"))
    return max(
        0.0,
        0.0 if not model_group_info.rpm or requests is None else int(str(requests)) / model_group_info.rpm,
        0.0 if not model_group_info.tpm or tokens is None else float(str(tokens)) / model_group_info.tpm,
    )


@dataclass(frozen=True, slots=True)
class _Seed:
    counter: RateLimitCounter
    value: int
    held: LocalValue
    window: LocalValue | None


def _seed(rate_limits: RateLimitStore, counter: RateLimitCounter, value: int, window_size: int, now: float) -> _Seed:
    if rate_limits.local((counter.window,))[0] is None:
        rate_limits.set_local(counter.window, str(int(now)), window_size)
    held: Final = rate_limits.local((counter,))[0]
    rate_limits.set_local(counter, value, window_size)
    return _Seed(
        counter=counter,
        value=value,
        held=held if isinstance(held, int | float) else 0,
        window=rate_limits.local((counter.window,))[0],
    )


def _restore(rate_limits: RateLimitStore, seed: _Seed, window_size: int) -> None:
    now: Final = rate_limits.local((seed.counter,))[0]
    after: Final = now if isinstance(now, int | float) else seed.value
    renewed: Final = rate_limits.local((seed.counter.window,))[0] != seed.window
    rate_limits.set_local(seed.counter, after if renewed else seed.held + after - seed.value, window_size)


def _saturation_seeds(
    handler: "NativeDynamicRateLimitHandler", model: str, saturation: float
) -> tuple[tuple[RateLimitCounter, int], ...]:
    group: Final = handler.llm_router.registry().group(model)
    assert group is not None
    scope: Final = RateLimitScope("model_saturation_check", model)
    tokens: Final = ((scope.counter("tokens"), min(round(saturation * group.tpm), group.tpm - 1)),) if group.tpm else ()
    requests: Final = (
        ((scope.counter("requests"), 0 if group.tpm else min(round(saturation * group.rpm), group.rpm - 1)),)
        if group.rpm
        else ()
    )
    return tokens + requests


def seed_counter(handler: "NativeDynamicRateLimitHandler", counter: RateLimitCounter, value: int) -> None:
    _seed(handler.rate_limits, counter, value, handler.window_size, handler.now())


def counter_count(handler: "NativeDynamicRateLimitHandler", counter: RateLimitCounter) -> LocalValue | None:
    return handler.rate_limits.local((counter,))[0]


async def pre_call_at_saturation(
    handler: "NativeDynamicRateLimitHandler",
    user_api_key_dict: UserAPIKeyAuth,
    model: str,
    saturation: float,
    data: dict[str, object] | None = None,  # mutable-ok: the pre-call hook rewrites the request dict in place
) -> None:
    rate_limits: Final = handler.rate_limits
    window_size: Final = handler.window_size
    request: Final = data if data is not None else {"max_tokens": 0}
    requested: Final = request.get("model")
    request["model"] = model
    now: Final = handler.now()
    seeds: Final = tuple(
        _seed(rate_limits, counter, value, window_size, now)
        for counter, value in _saturation_seeds(handler, model, saturation)
    )
    try:
        await handler.async_pre_call_hook(user_api_key_dict, DualCache(), request, "acompletion")
    finally:
        for seed in seeds:
            _restore(rate_limits, seed, window_size)
        if requested is None:
            del request["model"]
        else:
            request["model"] = requested


def _published_priority_descriptors(window_size: int) -> list[RateLimitDescriptor]:
    stash: Final = get_request_stash()
    response: Final = None if stash is None else stash.rate_limit_response
    statuses: Final = () if response is None else response["statuses"]
    values: Final = tuple(
        dict.fromkeys(
            status.get("descriptor_value") for status in statuses if status["descriptor_key"] == "priority_model"
        )
    )
    return [
        RateLimitDescriptor(
            key="priority_model",
            value=str(value),
            rate_limit={
                "requests_per_unit": next(
                    (
                        status["current_limit"]
                        for status in statuses
                        if status.get("descriptor_value") == value and status["rate_limit_type"] == "requests"
                    ),
                    None,
                ),
                "tokens_per_unit": next(
                    (
                        status["current_limit"]
                        for status in statuses
                        if status.get("descriptor_value") == value and status["rate_limit_type"] == "tokens"
                    ),
                    None,
                ),
                "window_size": window_size,
            },
        )
        for value in values
    ]


async def priority_based_descriptors(
    handler: "NativeDynamicRateLimitHandler", model: str, user_api_key_dict: UserAPIKeyAuth, priority: str | None
) -> list[RateLimitDescriptor]:
    assert (user_api_key_dict.metadata or {}).get("priority") == priority
    await handler.async_pre_call_hook(user_api_key_dict, DualCache(), {"model": model, "max_tokens": 0}, "acompletion")
    return _published_priority_descriptors(handler.window_size)


async def _no_legacy(_logger: object, _data: object) -> None:
    return None


def _call(data: Mapping[str, object], call_type: str) -> CallFacts:
    return CallFacts(
        ids=CallIds(call_id=str(data.get("litellm_call_id") or "")),
        kind=CallKind(call_type=call_type),
        target=CallTarget(model=str(data.get("model"))),
        timing=CallTiming(start=1.0),
    )


class NativeDynamicRateLimitHandler:
    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        build: Callable[..., NativeDynamicRateLimiter] = build_native_dynamic_rate_limiter,
    ) -> None:
        self._internal_usage_cache: Final = internal_usage_cache
        self._time_provider: Final = time_provider
        self._build: Final = build
        self.logger: NativeDynamicRateLimiter = build(internal_usage_cache, time_provider=time_provider)
        self.callbacks: list[object] = [self.logger]

    @property
    def rate_limits(self) -> RateLimitStore:
        return self.logger.rate_limits

    @property
    def window_size(self) -> int:
        return self.logger.window_size

    def now(self) -> float:
        return (self._time_provider or datetime.now)().timestamp()

    @property
    def v3_limiter(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return self.logger.v3_limiter

    @property
    def internal_usage_cache(self) -> InternalUsageCache:
        return self.logger.v3_limiter.internal_usage_cache

    @property
    def active_teams(self) -> ActiveTeamRegistry:
        return self.logger.delegate.active_teams

    @property
    def llm_router(self) -> Router:
        router: Final = self.logger.delegate.llm_router
        assert router is not None
        return router

    @v3_limiter.setter
    def v3_limiter(self, limiter: PROXY_MaxParallelRequestsHandler_v3) -> None:
        router: Final = self.logger.delegate.llm_router
        self.logger = self._build(  # rebind-ok: the engine is built on the store the test hands in
            self._internal_usage_cache, time_provider=self._time_provider, rate_limits=limiter.rate_limits
        )
        self.callbacks = [self.logger]  # rebind-ok: the rebuilt logger takes the list position
        vars(self.logger.delegate)["v3_limiter"] = limiter
        if router is not None:
            self.logger.update_variables(router)

    @active_teams.setter
    def active_teams(self, registry: ActiveTeamRegistry) -> None:
        vars(self.logger.delegate)["active_teams"] = registry

    def update_variables(self, llm_router: Router) -> None:
        self.logger.update_variables(llm_router)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: the legacy pre-call hook rewrites the proxy's request dict
        call_type: str,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            result: Final = await run_subscribers(
                hook_subscribers(Hook.pre_call, self.callbacks),
                with_sources(
                    KwargsFields(data, PROXY_REQUEST_KEYS),
                    PRE_CALL_SOURCES,
                    HookArgs(auth=user_api_key_dict, request=data, call_type=call_type),
                ),
                _no_legacy,
                call=_call(data, call_type),
            )
            after: Final = kwargs_after(raise_v2_abort(result))
        finally:
            REQUEST_AUTH.reset(auth_token)
        if isinstance(after, dict) and after is not data:
            data.clear()
            data.update(after)

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_success, self.callbacks),
                with_sources(
                    hook_fields(data, HookRun(response=response), Field.response),
                    DERIVED_POST_CALL_SUCCESS_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)
        return response

    async def async_post_call_response_headers_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, str] | None:  # mutable-ok: the legacy hook returns the headers it adds
        merged: Final[dict[str, str]] = {}
        await run_subscribers(
            hook_subscribers(Hook.response_headers, self.callbacks, admit=Admit.every_entry),
            with_sources(
                hook_fields(data, HookRun(response=merged), Field.response_headers),
                DERIVED_RESPONSE_HEADERS_FIELDS,
                HookArgs(auth=user_api_key_dict, request=data),
            ),
            _no_legacy,
        )
        return merged or None

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the native runs on the proxy's request dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_failure, self.callbacks, admit=Admit.every_entry),
                with_sources(
                    hook_fields(request_data, HookRun(response=original_exception), Field.error),
                    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=request_data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def _attempt_ended(self, kwargs: dict[str, object], response_obj: object, failed: bool) -> None:
        call_type: Final = kwargs.get("call_type")
        await run_subscribers(
            hook_subscribers(Hook.attempt_ended, self.callbacks, admit=Admit.every_entry),
            with_sources(
                KwargsFields(kwargs, LOGGING_KEYS),
                DERIVED_ATTEMPT_FIELDS,
                HookArgs(
                    request=kwargs,
                    result=response_obj,
                    outcome=AttemptOutcome.failure if failed else AttemptOutcome.success,
                ),
            ),
            _no_legacy,
            call=_call(kwargs, call_type if isinstance(call_type, str) else "acompletion"),
        )

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=False)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=True)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        return await self.logger.async_fallback_target_allowed(user_api_key_dict, model, request_data, commit=commit)

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        await self.logger.enforce_token_limits_for_frame(
            user_api_key_dict=user_api_key_dict, requested_model=requested_model
        )


def _resp_command(stream: "socket.SocketIO") -> list[bytes] | None:
    header: Final = stream.readline()
    if not header.startswith(b"*"):
        return None
    parts: Final[list[bytes]] = []
    for _ in range(int(header[1:])):
        size = int(stream.readline()[1:])
        parts.append(stream.read(size + 2)[:-2])
    return parts


def _bulk(text: str | None) -> bytes:
    if text is None:
        return b"$-1\r\n"
    return b"$" + str(len(text.encode())).encode() + b"\r\n" + text.encode() + b"\r\n"


def _resp_reply(command: list[bytes], counts: Mapping[str, str]) -> bytes:
    name: Final = command[0].upper()
    if name in (b"EVALSHA", b"EVAL"):
        keys: Final = [key.decode() for key in command[3 : 3 + int(command[2])]]
        return b"*" + str(len(keys)).encode() + b"\r\n" + b"".join(_bulk(counts.get(key)) for key in keys)
    if name == b"PING":
        return b"+PONG\r\n"
    if name == b"SCRIPT":
        return _bulk("0" * 40)
    return b"+OK\r\n"


@contextmanager
def redis_answering_counts(counts: Mapping[str, str]) -> Iterator[int]:
    listener: Final = socket.create_server(("127.0.0.1", 0))
    stopped: Final = threading.Event()

    def answer(connection: socket.socket) -> None:
        with connection, connection.makefile("rb") as stream:
            while (command := _resp_command(stream)) is not None:
                connection.sendall(_resp_reply(command, counts))

    def serve() -> None:
        while not stopped.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=answer, args=(connection,), daemon=True).start()

    server: Final = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        yield listener.getsockname()[1]
    finally:
        stopped.set()
        listener.close()
        server.join(timeout=5)
