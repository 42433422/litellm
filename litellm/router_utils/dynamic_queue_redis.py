"""The ``dynamic_queue`` contract shared by every worker through Redis.

Each operation is one Lua script call over five keys sharing a hash tag, so it is atomic and lands on one Redis
Cluster slot. Time is Redis ``TIME``, the one clock every worker agrees on. Waiters sort by member text, so the order
is exact for any integer priority and FIFO by an atomic sequence within one, with no floating-point score.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.router_utils.dynamic_queue import (
    AdmitReply,
    CoordinationUnavailable,
    Gone,
    Grant,
    MatchingFailure,
    Outcome,
    PassThrough,
    PollReply,
    Probe,
    QueueFull,
    QueueSettings,
    Released,
    Succeeded,
    Waiting,
    WaitRequest,
)

_RANK_WIDTH: Final = 6
_SEQ_WIDTH: Final = 20

_SCRIPT: Final = """
local state, queue, members, expiry, grants = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local op, waiter = ARGV[1], ARGV[3]
local now
if ARGV[2] == '' then
  local t = redis.call('TIME')
  now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
else
  now = tonumber(ARGV[2])
end
local required, initial_backoff, max_backoff = tonumber(ARGV[4]), tonumber(ARGV[5]), tonumber(ARGV[6])
local lease_ms, liveness_ms, max_waiting = tonumber(ARGV[7]), tonumber(ARGV[8]), tonumber(ARGV[9])
local ttl_ms, sweep_limit = tonumber(ARGV[10]), tonumber(ARGV[11])

local function num(name)
  return tonumber(redis.call('HGET', state, name) or '0')
end

local function waiter_of(member)
  return string.sub(member, string.find(member, '|', 1, true) + 22)
end

local function clear_lease()
  redis.call('HDEL', state, 'lease_waiter', 'lease_token', 'lease_until')
end

local function drop(w)
  local entry = redis.call('HGET', members, w)
  if entry then
    redis.call('ZREM', queue, string.sub(entry, string.find(entry, '|', 1, true) + 1))
    redis.call('HDEL', members, w)
  end
  redis.call('ZREM', expiry, w)
  redis.call('HDEL', grants, w)
  if redis.call('HGET', state, 'lease_waiter') == w then
    clear_lease()
  end
end

local function sweep()
  if redis.call('HEXISTS', state, 'lease_until') == 1 and num('lease_until') <= now then
    clear_lease()
  end
  for _, w in ipairs(redis.call('ZRANGEBYSCORE', expiry, '-inf', now, 'LIMIT', 0, sweep_limit)) do
    drop(w)
  end
end

local function congested()
  return redis.call('HGET', state, 'mode') == 'congested'
end

local function grant_probe()
  if not congested() or redis.call('HEXISTS', state, 'lease_waiter') == 1 or now < num('next_probe') then
    return
  end
  local head = redis.call('ZRANGE', queue, 0, 0)[1]
  if not head then
    return
  end
  local w = waiter_of(head)
  local token = redis.call('HINCRBY', state, 'tokens', 1)
  redis.call('HSET', state, 'lease_waiter', w, 'lease_token', token, 'lease_until', now + lease_ms)
  redis.call('ZREM', queue, head)
  redis.call('HDEL', members, w)
  redis.call('HSET', grants, w, 'probe|' .. num('generation') .. '|' .. token)
end

local function take_grant(w)
  local grant = redis.call('HGET', grants, w)
  if grant then
    redis.call('HDEL', grants, w)
    redis.call('ZREM', expiry, w)
  end
  return grant
end

local function congest(retry_after, failed_probe)
  local backoff = initial_backoff
  if congested() then
    backoff = num('backoff')
    if failed_probe then
      backoff = math.min(backoff * 2, max_backoff)
    end
  end
  redis.call('HINCRBY', state, 'generation', 1)
  clear_lease()
  local next_probe = math.max(num('next_probe'), now + math.max(backoff, retry_after))
  redis.call('HSET', state, 'mode', 'congested', 'successes', 0, 'backoff', backoff, 'next_probe', next_probe)
end

local function recover()
  local generation = redis.call('HINCRBY', state, 'generation', 1)
  redis.call('HDEL', state, 'mode', 'successes', 'backoff', 'next_probe')
  clear_lease()
  for order, member in ipairs(redis.call('ZRANGE', queue, 0, -1)) do
    local w = waiter_of(member)
    redis.call('HSET', grants, w, 'released|' .. generation .. '|' .. order)
    redis.call('HDEL', members, w)
  end
  redis.call('DEL', queue)
end

local function touch()
  for _, key in ipairs(KEYS) do
    redis.call('PEXPIRE', key, ttl_ms)
  end
end

local function settle(w)
  grant_probe()
  local grant = take_grant(w)
  touch()
  if grant then
    return {'grant', grant}
  end
  return {'wait'}
end

sweep()

if op == 'admit' then
  if not congested() then
    touch()
    return {'pass', num('generation')}
  end
  if redis.call('ZCARD', queue) >= max_waiting then
    touch()
    return {'full'}
  end
  local seq = redis.call('HINCRBY', state, 'seq', 1)
  local member = ARGV[12] .. '|' .. string.format('%020d', seq) .. '|' .. waiter
  local deadline = now + tonumber(ARGV[13])
  redis.call('ZADD', queue, 0, member)
  redis.call('HSET', members, waiter, deadline .. '|' .. member)
  redis.call('ZADD', expiry, math.min(deadline, now + liveness_ms), waiter)
  return settle(waiter)
end

if op == 'poll' then
  if redis.call('HEXISTS', grants, waiter) == 0 then
    local entry = redis.call('HGET', members, waiter)
    if not entry then
      return {'gone'}
    end
    local deadline = tonumber(string.sub(entry, 1, string.find(entry, '|', 1, true) - 1))
    redis.call('ZADD', expiry, math.min(deadline, now + liveness_ms), waiter)
  end
  return settle(waiter)
end

if op == 'renew' then
  if redis.call('HGET', state, 'lease_waiter') == waiter and num('lease_token') == tonumber(ARGV[12]) then
    redis.call('HSET', state, 'lease_until', now + lease_ms)
    touch()
    return {'renewed'}
  end
  return {'lost'}
end

if op == 'report' then
  local active_probe = ARGV[12] == 'probe'
    and redis.call('HGET', state, 'lease_waiter') == waiter
    and num('lease_token') == tonumber(ARGV[14])
    and num('generation') == tonumber(ARGV[13])
  if ARGV[15] == 'matching' then
    congest(tonumber(ARGV[16]), active_probe)
  elseif active_probe then
    clear_lease()
    if ARGV[15] == 'success' then
      if redis.call('HINCRBY', state, 'successes', 1) >= required then
        recover()
      end
    else
      redis.call('HSET', state, 'successes', 0)
    end
    grant_probe()
  end
  touch()
  return {'ok'}
end

if op == 'abandon' then
  drop(waiter)
  grant_probe()
  touch()
  return {'ok'}
end

return redis.error_reply('unknown dynamic queue op ' .. op)
"""

_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


class ScriptHost(Protocol):
    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]: ...


def rank_key(priority: int) -> str:
    """Text that sorts byte-wise in the same order as ``priority`` does numerically, for any integer."""
    digits: Final = str(abs(priority))
    if priority >= 0:
        return f"p{len(digits):0{_RANK_WIDTH}d}{digits}"
    complement: Final = digits.translate(str.maketrans("0123456789", "9876543210"))
    return f"n{10**_RANK_WIDTH - 1 - len(digits):0{_RANK_WIDTH}d}{complement}"


def _keys(capacity: str) -> tuple[str, ...]:
    tag: Final = f"litellm:dynamic_queue:{{dq:{capacity}}}"
    return tuple(f"{tag}:{part}" for part in ("state", "queue", "members", "expiry", "grants"))


def _text(item: str | bytes | int) -> str:
    return item.decode() if isinstance(item, bytes) else str(item)


def _grant(encoded: str) -> Grant | CoordinationUnavailable:
    match tuple(encoded.split("|")):
        case ("probe", generation, token):
            return Probe(generation=int(generation), token=int(token))
        case ("released", generation, order):
            return Released(generation=int(generation), order=int(order))
        case _:
            return CoordinationUnavailable(reason=f"unexpected grant {encoded!r}")


def _settled(reply: tuple[str, ...]) -> Grant | Waiting | QueueFull | Gone | CoordinationUnavailable:
    match reply:
        case ("pass", generation):
            return PassThrough(generation=int(generation))
        case ("grant", encoded):
            return _grant(encoded)
        case ("wait",):
            return Waiting()
        case ("full",):
            return QueueFull()
        case ("gone",):
            return Gone()
        case _:
            return CoordinationUnavailable(reason=f"unexpected reply {reply!r}")


def _outcome_args(outcome: Outcome) -> tuple[str, int]:
    if isinstance(outcome, MatchingFailure):
        return "matching", outcome.retry_after_ms
    return ("success" if isinstance(outcome, Succeeded) else "other"), 0


def _grant_args(grant: Grant) -> tuple[str, int, int]:
    match grant:
        case Probe(generation=generation, token=token):
            return "probe", generation, token
        case Released(generation=generation):
            return "released", generation, 0
        case PassThrough(generation=generation):
            return "pass", generation, 0


class RedisQueueBackend:
    """``clock`` stands in for Redis ``TIME`` with milliseconds of the caller's choosing, so a test can drive every
    worker's time together. Production leaves it unset."""

    def __init__(
        self,
        scripts: ScriptHost,
        settings: QueueSettings | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._run = scripts.async_register_script(_SCRIPT)
        self._settings = settings or QueueSettings()
        self._clock = clock

    @property
    def settings(self) -> QueueSettings:
        return self._settings

    async def _call(
        self, op: str, capacity: str, waiter_id: str, extra: Sequence[str | int]
    ) -> tuple[str, ...] | CoordinationUnavailable:
        from redis.exceptions import RedisError

        settings: Final = self._settings
        args: Final = (
            op,
            "" if self._clock is None else self._clock(),
            waiter_id,
            settings.required_successes,
            settings.initial_backoff_ms,
            settings.max_backoff_ms,
            settings.lease_ms,
            settings.liveness_ms,
            settings.max_waiting,
            settings.state_ttl_ms,
            settings.sweep_limit,
            *extra,
        )
        try:
            raw: Final = await self._run(keys=_keys(capacity), args=args)
            return tuple(_text(item) for item in _REPLY.validate_python(raw))
        except (RedisError, RedisCircuitBreakerOpenError, OSError, ValidationError) as error:
            return CoordinationUnavailable(reason=f"{type(error).__name__}: {error}")

    async def admit(self, request: WaitRequest, budget_ms: int) -> AdmitReply:
        reply: Final = await self._call(
            "admit", request.capacity, request.waiter_id, (rank_key(request.priority), budget_ms)
        )
        if isinstance(reply, CoordinationUnavailable):
            return reply
        settled: Final = _settled(reply)
        if isinstance(settled, Gone):
            return CoordinationUnavailable(reason=f"unexpected admit reply {reply!r}")
        return settled

    async def poll(self, capacity: str, waiter_id: str) -> PollReply:
        reply: Final = await self._call("poll", capacity, waiter_id, ())
        if isinstance(reply, CoordinationUnavailable):
            return reply
        settled: Final = _settled(reply)
        if isinstance(settled, QueueFull):
            return CoordinationUnavailable(reason=f"unexpected poll reply {reply!r}")
        return settled

    async def renew(self, capacity: str, waiter_id: str, token: int) -> bool | CoordinationUnavailable:
        reply: Final = await self._call("renew", capacity, waiter_id, (token,))
        return reply if isinstance(reply, CoordinationUnavailable) else reply == ("renewed",)

    async def report(
        self, capacity: str, waiter_id: str, grant: Grant, outcome: Outcome
    ) -> None | CoordinationUnavailable:
        reply: Final = await self._call("report", capacity, waiter_id, (*_grant_args(grant), *_outcome_args(outcome)))
        return reply if isinstance(reply, CoordinationUnavailable) else None

    async def abandon(self, capacity: str, waiter_id: str) -> None | CoordinationUnavailable:
        reply: Final = await self._call("abandon", capacity, waiter_id, ())
        return reply if isinstance(reply, CoordinationUnavailable) else None
