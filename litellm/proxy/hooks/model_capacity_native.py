import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from weakref import WeakKeyDictionary

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm._internal_context import service_target
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.exceptions import ServiceUnavailableError
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import key_identity, mapping, member
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import (
    RustHandle,
    RustLogger,
    WireAbort,
    WireManifest,
    _namespace,  # pyright: ignore[reportPrivateUsage]  # the generated class needs this port's base
    register_abort_error,
)
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError, map_v3_rate_limit_type
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RATE_LIMIT_UNVERIFIABLE_MESSAGE,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitedModel,
    RateLimitUnverifiableError,
    RequestRateLimiterStash,
    borrowing_team_model_limits,
    callback_kwargs_mapping,
    claim_request_stash_for_data,
    get_or_create_request_stash,
    get_request_stash,
    native_rate_limit_store,
    request_stash_for_callback,
)
from litellm.proxy.hooks.parallel_request_limiter_v3_native import policy as limiter_policy
from litellm.proxy.hooks.parallel_request_limiter_v3_native import pre_call_request
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, RateLimitStore
from litellm.types.integrations.custom_logger_v2 import KeyIdentity
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome
from litellm.types.utils import PriorityReservationSettings

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.rust_bridge._native import NativeModelCapacity, NativeModelCapacityState

UNLICENSED_PRIORITY_MESSAGE: Final = (
    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your "
    ".env to enable this.\nGet a license: https://docs.litellm.ai/docs/enterprise."
)
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_TRACES: Final = frozenset({"untraced", "attempt", "key"})
RATE_LIMIT_KIND: Final = "model_capacity_rate_limit"
RERAISE_KIND: Final = "model_capacity_reraise"
_TOKENS: Final = count(1)
_LOG_LEVELS: Final = MappingProxyType({"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO})
_TARGETED_HOOKS: Final = frozenset({Hook.pre_call, Hook.post_call_success, Hook.post_call_failure})


@dataclass(slots=True)
class LiveRequest:
    hook: Hook | None = None
    data: Mapping[str, object] | None = None
    auth: UserAPIKeyAuth | None = None
    response: object = None
    kwargs: Mapping[str, object] | None = None
    state: "NativeModelCapacityState | None" = None
    raised: dict[int, BaseException] = field(default_factory=dict)  # mutable-ok: exceptions the seam re-raises


_LIVE: Final[ContextVar[LiveRequest | None]] = ContextVar("native_model_capacity_live", default=None)


_ADMISSIONS: Final[WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]] = WeakKeyDictionary()


def _admission(limiter: PROXY_MaxParallelRequestsHandler_v3) -> AbstractAsyncContextManager[object]:
    if limiter.rate_limits.has_redis:
        return nullcontext()
    loop: Final = asyncio.get_running_loop()
    lock: Final = _ADMISSIONS.get(loop)
    if lock is not None:
        return lock
    created: Final = asyncio.Lock()
    _ADMISSIONS[loop] = created
    return created


def _live() -> LiveRequest:
    live: Final = _LIVE.get()
    if live is None:
        raise RuntimeError("the model capacity seam was called outside a live request")
    return live


def auth_from_request(data: object) -> UserAPIKeyAuth | None:
    request: Final = as_str_mapping(data)
    if request is None:
        return None
    for bucket_name in _METADATA_BUCKETS:
        bucket = as_str_mapping(request.get(bucket_name))
        auth = None if bucket is None else bucket.get("user_api_key_auth")
        if isinstance(auth, UserAPIKeyAuth):
            return auth
        if as_str_mapping(auth) is not None:
            try:
                return UserAPIKeyAuth.model_validate(auth)
            except ValidationError:
                continue
    return None


def _unheld(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return value
    return None


def key_gaps(auth: UserAPIKeyAuth | None) -> Mapping[str, object]:
    if auth is None:
        return {}
    return {
        "team_priority": _unheld(member(mapping(auth.team_metadata), "priority")),
        "key_priority": _unheld(member(mapping(auth.metadata), "priority")),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the key's untyped metadata dict
    }


def _dumped(record: BaseModel | None) -> object:
    return None if record is None else record.model_dump(mode="python")


def key_inputs(auth: UserAPIKeyAuth) -> Mapping[str, object]:
    args: Final = HookArgs(auth=auth)
    return {"identity": _dumped(key_identity(args)), "gaps": key_gaps(auth)}


class _SettlementMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_api_key_auth_metadata: Mapping[str, object] | None = None
    user_api_key_team_id: str | None = None
    user_api_key_team_alias: str | None = None


class _SettlementLoggingObject(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: _SettlementMetadata | None = None


def _prometheus_loggers() -> "Sequence[PrometheusLogger]":
    from litellm.integrations.prometheus import PrometheusLogger

    return tuple(
        logger
        for logger in litellm.logging_callback_manager.get_custom_loggers_for_type(PrometheusLogger)
        if isinstance(logger, PrometheusLogger)
    )


def _settlement_error(kwargs: Mapping[str, object]) -> str | None:
    try:
        _SettlementLoggingObject.model_validate(kwargs.get("standard_logging_object") or {})
    except ValidationError as error:
        return str(error)
    return None


def _attempt_gaps(raw: object) -> Mapping[str, object]:
    return {"settlement_error": _settlement_error(callback_kwargs_mapping(raw))}


def _project_kwargs(raw: object) -> object:
    live: Final = _LIVE.get()
    match None if live is None else live.hook:
        case Hook.pre_call:
            return key_gaps(REQUEST_AUTH.get())
        case Hook.attempt_ended:
            return _attempt_gaps(raw)
        case _:
            return None


PROJECTIONS: Final[Mapping[Field, Callable[[object], object]]] = {
    Field.deprecated_kwargs: _project_kwargs,
    Field.deprecated_response_obj: lambda _response: None,
}


_ARGS: Final = TypeAdapter(dict[str, object])
_PAIRS: Final = TypeAdapter(tuple[tuple[str, object], ...])


def _registry_unavailable(error: Exception) -> bool:
    if isinstance(error, RedisCircuitBreakerOpenError | ValidationError):
        return True
    try:
        from redis.exceptions import RedisError
    except ImportError:
        return False
    return isinstance(error, RedisError)


def _raised(error: Exception) -> Mapping[str, object]:
    live: Final = _LIVE.get()
    token: Final = next(_TOKENS)
    if live is not None:
        live.raised[token] = error
    kind: Final = (
        "http"
        if isinstance(error, HTTPException)
        else "registry_unavailable"
        if _registry_unavailable(error)
        else "other"
    )
    return {"raised": {"token": token, "kind": kind, "message": str(error)}}


def _seam(
    method: Callable[["CapacityDelegate", Mapping[str, object]], Awaitable[object]],
) -> Callable[["CapacityDelegate", object], Awaitable[Mapping[str, object]]]:
    async def call(self: "CapacityDelegate", args: object) -> Mapping[str, object]:
        try:
            return {"ok": await method(self, _ARGS.validate_python(args))}
        except Exception as error:  # noqa: BLE001  # Rust decides which errors fail open and which re-raise
            return _raised(error)

    return call


class CapacityDelegate:
    def __init__(
        self,
        internal_usage_cache: DualCache,
        v3_limiter: PROXY_MaxParallelRequestsHandler_v3,
    ) -> None:
        self.v3_limiter: Final = v3_limiter
        self.active_teams: Final = ActiveTeamRegistry(internal_usage_cache)
        self.llm_router: Router | None = None

    def _router(self) -> Router:
        if self.llm_router is None:
            raise AttributeError("'CapacityDelegate' object has no attribute 'llm_router'")
        return self.llm_router

    def _auth(self) -> UserAPIKeyAuth:
        auth: Final = _live().auth
        if auth is None:
            raise ValueError("the request carries no key auth")
        return auth

    @_seam
    async def settings(self, _args: Mapping[str, object]) -> object:
        settings: Final = litellm.priority_reservation_settings or PriorityReservationSettings()
        reservation: Final = litellm.priority_reservation
        return {
            "priority_reservation": None
            if reservation is None
            else [[name, value] for name, value in reservation.items()],
            "licensed": os.getenv("LITELLM_LICENSE", None) is not None,
            "default_priority": settings.default_priority,
            "saturation_threshold": settings.saturation_threshold,
            "class_split": settings.priority_class_share_splits_equally_among_active_teams,
            "saturation_check_cache_ttl": settings.saturation_check_cache_ttl,
            "window_size": self.v3_limiter.window_size,
            "rate_limit_type": self.v3_limiter.get_rate_limit_type(),
            "policy": limiter_policy(self.v3_limiter),
        }

    @_seam
    async def group(self, args: Mapping[str, object]) -> object:
        group: Final = self._router().registry().group(str(args["model"]))
        if group is None:
            return None
        return {
            "tpm": group.tpm,
            "rpm": group.rpm,
            "input_cost_per_token": group.input_cost_per_token,
            "output_cost_per_token": group.output_cost_per_token,
        }

    @_seam
    async def borrowing_limits(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        rate_limited: Final = self.v3_limiter.rate_limited_model(model) or RateLimitedModel(
            requested=model, group=model
        )
        limits: Final = borrowing_team_model_limits(self._auth(), rate_limited)
        if limits.tpm is None and limits.rpm is None:
            return None
        return {
            "group": rate_limited.group,
            "tpm": limits.tpm,
            "rpm": limits.rpm,
            "max_borrow_multiple": limits.max_borrow_multiple,
        }

    @_seam
    async def model_group(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        return (self.v3_limiter.rate_limited_model(model) or RateLimitedModel(requested=model, group=model)).group

    @_seam
    async def estimate_tokens(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        data: Final = live.data
        if not data:
            return {"kind": "no_body"}
        model: Final = str(args["model"])
        tpm_share: Final = int(str(args["tpm_share"]))
        call_type: Final = args["call_type"] if isinstance(args["call_type"], str) else None
        as_dispatched: Final = dict(data, model=model)
        if self.v3_limiter.get_rate_limit_type() != "input":
            self.v3_limiter.bound_output_for_limit(as_dispatched, tpm_share, call_type)
        try:
            estimate: Final = await offload_token_count(self.v3_limiter.estimate_request_tokens)(
                data=as_dispatched,
                model=model,
                min_configured_tpm_limit=tpm_share,
                call_type=call_type,
                team_id=None if live.auth is None else live.auth.team_id,
                rate_limit_type=self.v3_limiter.get_rate_limit_type(),
            )
        except Exception as error:  # noqa: BLE001  # an unreadable prompt still holds the minimum so admission runs
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.warning(
                "Dynamic rate limiter: token estimate failed for %s, holding one token: %s", model, error
            )
            return {"kind": "failed"}
        return {"kind": "tokens", "tokens": estimate}

    @_seam
    async def cap_output(self, args: Mapping[str, object]) -> object:
        data: Final = _live().data
        call_type: Final = args["call_type"] if isinstance(args["call_type"], str) else None
        self.v3_limiter.bound_output_for_limit(data, int(str(args["tpm_share"])), call_type)
        return None

    @_seam
    async def stash_call_type(self, _args: Mapping[str, object]) -> object:
        return get_or_create_request_stash().call_type

    @_seam
    async def post_call_success(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        await self.v3_limiter.async_post_call_success_hook(  # pyright: ignore[reportUnknownMemberType]  # the legacy hook signature
            data=dict(live.data or {}) if not isinstance(live.data, dict) else live.data,
            user_api_key_dict=self._auth(),
            response=live.response,  # pyright: ignore[reportArgumentType]  # the live response the proxy handed this hook
        )
        if not response_has_hidden_params(live.response):
            return None
        additional_headers: Final = ensure_response_additional_headers(live.response)
        for name, value in _PAIRS.validate_python(args["headers"]):
            additional_headers[name] = value  # pyright: ignore[reportArgumentType]  # the priority value as the handler set it
        return None

    @_seam
    async def touch(self, args: Mapping[str, object]) -> object:
        members: Final = await self.active_teams.touch(
            str(args["model"]),
            str(args["member"]),
            args["priority_class"],  # pyright: ignore[reportArgumentType]  # the class as the handler passes it, a reserved priority of any type
            int(str(args["window"])),
        )
        return [[member, priority_class] for member, priority_class in members.items()]

    @_seam
    async def metrics_enabled(self, _args: Mapping[str, object]) -> object:
        return bool(_prometheus_loggers())

    @_seam
    async def throttled(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        for logger in _prometheus_loggers():
            try:
                logger.increment_dynamic_limiter_throttled(
                    model_group=str(args["model_group"]),
                    team=str(args["team"]),
                    team_alias=args["team_alias"] if isinstance(args["team_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority header value as the handler passes it
                    rate_limit_type=str(args["rate_limit_type"]),
                    reason=str(args["reason"]),
                )
            except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change the 429
                verbose_proxy_logger.warning("Dynamic rate limiter throttle metric failed: %s", metrics_error)
        return None

    @_seam
    async def log(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.log(_LOG_LEVELS[str(args["level"])], str(args["message"]))
        return None

    @_seam
    async def unlicensed_priority(self, _args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.error(UNLICENSED_PRIORITY_MESSAGE)
        return None

    @_seam
    async def set_share(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        try:
            for logger in _prometheus_loggers():
                logger.set_dynamic_limiter_share(
                    model_group=str(args["model_group"]),
                    team=str(args["team"]),
                    team_alias=args["team_alias"] if isinstance(args["team_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority value as the handler passes it
                    share_tokens=int(str(args["share_tokens"])),
                    borrowed_tokens=int(str(args["borrowed_tokens"])),
                )
        except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change settlement
            verbose_proxy_logger.warning("Dynamic rate limiter share metrics failed: %s", metrics_error)
        return None


def shared_rate_limit_store(internal_usage_cache: DualCache) -> RateLimitStore:
    registered: Final = next(
        (
            callback.rate_limits
            for callback in litellm.callbacks
            if isinstance(callback, PROXY_MaxParallelRequestsHandler_v3)
            and callback.internal_usage_cache.dual_cache is internal_usage_cache
        ),
        None,
    )
    if registered is not None:
        return registered
    return native_rate_limit_store(internal_usage_cache)


def _rate_limit_error(wire: WireAbort) -> Exception:
    return _proxy_rate_limit_error((None if wire.error is None else as_str_mapping(wire.error.fields)) or {})


def _proxy_rate_limit_error(fields: Mapping[str, object]) -> ProxyRateLimitError:
    raw_type: Final = fields.get("rate_limit_type")
    model: Final = fields.get("model")
    resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model if isinstance(model, str) else None)
    descriptor_key: Final = fields.get("descriptor_key")
    return ProxyRateLimitError(
        detail=dict(_PAIRS.validate_python(fields.get("detail") or ())),
        headers=dict(_PAIRS.validate_python(fields.get("headers") or ())),
        rate_limit_type=map_v3_rate_limit_type(raw_type if isinstance(raw_type, str) else None),
        model=resolved_model,
        llm_provider=llm_provider,
        descriptor_key=descriptor_key if isinstance(descriptor_key, str) else None,
    )


def _reraise(wire: WireAbort) -> Exception:
    live: Final = _LIVE.get()
    fields: Final = (None if wire.error is None else as_str_mapping(wire.error.fields)) or {}
    if fields.get("unverifiable") is True:
        return RateLimitUnverifiableError()
    token: Final = fields.get("token")
    raised: Final = None if live is None or not isinstance(token, int) else live.raised.pop(token, None)
    if isinstance(raised, Exception):
        return raised
    return HTTPException(status_code=wire.status, detail={"error": wire.message}, headers=dict(wire.headers))


def _native_answer(answer: Mapping[str, object], live: LiveRequest) -> Mapping[str, object]:
    if answer.get("unverifiable") is True:
        raise RateLimitUnverifiableError()
    raised_token: Final = answer.get("raised")
    if isinstance(raised_token, int) and raised_token in live.raised:
        raise live.raised[raised_token]
    if "type_error" in answer:
        raise TypeError(str(answer["type_error"]))
    if "failed" in answer:
        raise RuntimeError(str(answer["failed"]))
    return answer


@dataclass(frozen=True, slots=True)
class _StashedNative:
    capacity: "NativeModelCapacity"
    callback: RustHandle

    def manifest(self) -> str:
        return self.callback.manifest()

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, _state: object, /
    ) -> Awaitable[object]:
        live: Final = _LIVE.get()
        request: Final = pre_call_request() if hook == Hook.pre_call.value else None
        return self.capacity.handle(hook, fields, call, None if live is None else live.state, request=request)


def _request_stash(hook: Hook, data: Mapping[str, object] | None) -> RequestRateLimiterStash | None:
    if hook is Hook.post_call_failure:
        return get_request_stash()
    if data is None:
        return None
    match hook:
        case Hook.pre_call:
            return claim_request_stash_for_data(data) if "model" in data else None
        case Hook.attempt_ended:
            return request_stash_for_callback(callback_kwargs_mapping(data))
        case _:
            return None


def _targeted(hook: Hook, view: object) -> bool:
    if hook is Hook.attempt_ended:
        return getattr(view, Field.outcome.value, None) == AttemptOutcome.success
    return hook in _TARGETED_HOOKS


class NativeDynamicRateLimiter(RustLogger):
    delegate: CapacityDelegate
    capacity: "NativeModelCapacity"
    rate_limits: RateLimitStore
    window_size: int

    def __new__(
        cls,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        *,
        rate_limits: RateLimitStore | None = None,
    ) -> "NativeDynamicRateLimiter":
        from litellm.rust_bridge._native import NativeModelCapacity, NativeRateLimits

        register_abort_error(RATE_LIMIT_KIND, _rate_limit_error)
        register_abort_error(RERAISE_KIND, _reraise)
        store: Final = rate_limits or shared_rate_limit_store(internal_usage_cache)
        v3_limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
            InternalUsageCache(dual_cache=internal_usage_cache),
            time_provider=time_provider,
            rate_limit_store=store,
        )
        handle: Final = store.handle
        if not isinstance(handle, NativeRateLimits):
            raise TypeError(f"model capacity counts on the native rate limit store, not {type(handle).__name__}")
        delegate: Final = CapacityDelegate(internal_usage_cache, v3_limiter)
        capacity: Final = NativeModelCapacity(
            delegate,
            handle,
            v3_limiter.window_size,
            v3_limiter.tpm_reservation_enabled,
            None if time_provider is None else (lambda: time_provider().timestamp()),
        )
        native: Final = _StashedNative(capacity, capacity.callback())
        manifest: Final = WireManifest.model_validate_json(native.manifest())
        logger: Final = object.__new__(type(manifest.selection.name, (cls,), _namespace(native, manifest)))
        RustLogger.__init__(logger, native, PROJECTIONS)
        logger.delegate = delegate
        logger.capacity = capacity
        logger.rate_limits = store
        logger.window_size = v3_limiter.window_size
        return logger

    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        *,
        rate_limits: RateLimitStore | None = None,
    ) -> None:
        pass

    @property
    def v3_limiter(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return self.delegate.v3_limiter

    def update_variables(self, llm_router: Router) -> None:
        self.delegate.llm_router = llm_router

    def _state(self, stash: RequestRateLimiterStash) -> "NativeModelCapacityState":
        return stash.native_capacity(self.capacity.request_state)

    def _state_for(self, stash: RequestRateLimiterStash | None) -> "NativeModelCapacityState | None":
        return None if stash is None else self._state(stash)

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        data: Final = as_str_mapping(getattr(view, Field.deprecated_kwargs.value, None))
        live: Final = LiveRequest(
            hook=hook,
            data=data if hook is not Hook.attempt_ended else None,
            auth=(REQUEST_AUTH.get() or auth_from_request(data)) if hook is not Hook.attempt_ended else None,
            response=getattr(view, Field.deprecated_response_obj.value, None),
            kwargs=data if hook is Hook.attempt_ended else None,
            state=self._state_for(_request_stash(hook, data)),
        )
        token: Final = _LIVE.set(live)
        try:
            with service_target(RATE_LIMITS_TARGET) if _targeted(hook, view) else nullcontext():
                if hook is Hook.post_call_success:
                    identity: Final = getattr(view, Field.key_identity.value, None)
                    headers: Final = self.capacity.post_call_headers(
                        {
                            "identity": _dumped(identity if isinstance(identity, KeyIdentity) else None),
                            "gaps": key_gaps(REQUEST_AUTH.get()),
                        }
                    )
                    answer: Final = await self.delegate.post_call_success({"headers": headers})
                    raised: Final = as_str_mapping(answer.get("raised"))
                    raised_token: Final = None if raised is None else raised.get("token")
                    error: Final = None if not isinstance(raised_token, int) else live.raised.pop(raised_token, None)
                    if error is not None:
                        from litellm._logging import verbose_proxy_logger

                        verbose_proxy_logger.error(
                            "Error in dynamic rate limiter v3 post-call hook: %s", error, exc_info=error
                        )
                    return None
                if hook is not Hook.pre_call:
                    return await super().run_rust(hook, reads, view, ctx)
                async with _admission(self.v3_limiter):
                    return await super().run_rust(hook, reads, view, ctx)
        finally:
            _LIVE.reset(token)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await self.run_rust(
                Hook.post_call_failure,
                frozenset(),
                None,
                CallContext(hook=Hook.post_call_failure, call=None, state=None),
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        state: Final = self._state(get_or_create_request_stash())
        live: Final = LiveRequest(data={**request_data, "model": model}, auth=user_api_key_dict, state=state)
        token: Final = _LIVE.set(live)
        try:
            async with _admission(self.v3_limiter):
                answer: Final = await self.capacity.fallback_target_allowed(
                    state, key_inputs(user_api_key_dict), model, commit, request=pre_call_request()
                )
        finally:
            _LIVE.reset(token)
        return _native_answer(answer, live).get("allowed") is True

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        if requested_model is None:
            return
        state: Final = self._state_for(get_request_stash())
        live: Final = LiveRequest(auth=user_api_key_dict, state=state)
        token: Final = _LIVE.set(live)
        try:
            async with _admission(self.v3_limiter):
                answer: Final = await self.capacity.enforce_frame(
                    state, key_inputs(user_api_key_dict), requested_model, request=pre_call_request()
                )
        finally:
            _LIVE.reset(token)
        if answer.get("unverifiable") is True:
            raise ServiceUnavailableError(
                message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model
            ) from RateLimitUnverifiableError()
        refused: Final = as_str_mapping(_native_answer(answer, live).get("refused"))
        if refused is not None:
            raise _proxy_rate_limit_error(refused)


def build_native_dynamic_rate_limiter(
    internal_usage_cache: DualCache,
    time_provider: Callable[[], datetime] | None = None,
    *,
    rate_limits: RateLimitStore | None = None,
) -> NativeDynamicRateLimiter:
    return NativeDynamicRateLimiter(internal_usage_cache, time_provider, rate_limits=rate_limits)


@dataclass(slots=True)
class _Registered:
    logger: NativeDynamicRateLimiter | None = None


_REGISTERED: Final = _Registered()


def registered_native_dynamic_rate_limiter() -> NativeDynamicRateLimiter | None:
    return _REGISTERED.logger


def native_dynamic_rate_limiter(internal_usage_cache: DualCache, llm_router: object) -> NativeDynamicRateLimiter:
    if _REGISTERED.logger is None:
        _REGISTERED.logger = build_native_dynamic_rate_limiter(internal_usage_cache)
    if isinstance(llm_router, Router):
        _REGISTERED.logger.update_variables(llm_router)
    return _REGISTERED.logger


__all__ = [
    "CapacityDelegate",
    "NativeDynamicRateLimiter",
    "build_native_dynamic_rate_limiter",
    "native_dynamic_rate_limiter",
    "registered_native_dynamic_rate_limiter",
    "shared_rate_limit_store",
]
