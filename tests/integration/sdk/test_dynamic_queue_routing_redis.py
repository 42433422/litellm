import asyncio
import json
import os
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from redis import Redis

import litellm
from litellm.router_utils.dynamic_queue_redis import _keys
from litellm.router_utils.dynamic_queue_router import (
    QUEUE_FULL_CODE,
    DynamicQueueFullError,
    DynamicQueueTimeoutError,
)

CHAT_PATH: Final = "/v1/chat/completions"
MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _rate_limited(retry_after_s: int | None) -> Reply:
    return Reply(
        status=429,
        body=json.dumps({"error": {"message": "rate limited", "type": "rate_limit_error"}}).encode(),
        headers={} if retry_after_s is None else {"retry-after": str(retry_after_s)},
    )


@dataclass(frozen=True, slots=True)
class Upstream:
    wire: Wire
    replies: SimpleQueue[Reply]

    def chat_requests(self) -> int:
        """The chat completion requests the upstream received so far, read once."""
        return sum(1 for request in self.wire.drain() if request.target.startswith(CHAT_PATH))


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    replies: Final[SimpleQueue[Reply]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        try:
            return replies.get_nowait()
        except Empty:
            return chat_reply(f"chatcmpl-{uuid.uuid4()}", "gpt-x", "served", stream=False)

    with wire_server(respond) as wire:
        yield Upstream(wire, replies)


def _worker(
    upstream: Upstream, capacity: str, max_parallel_requests: int | None = None, **queue: object
) -> litellm.Router:
    """One proxy worker's router: its own process state, the shared Redis, the same deployment."""
    slots: Final = {} if max_parallel_requests is None else {"max_parallel_requests": max_parallel_requests}
    return litellm.Router(
        model_list=[
            {
                "model_name": "group",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_base": f"{upstream.wire.url}/v1",
                    "api_key": "integration-key",
                    "max_retries": 0,
                    **slots,
                },
                "model_info": {"id": capacity},
            }
        ],
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
        default_priority=0,
        num_retries=0,
        cooldown_time=1,
        dynamic_queue={"enabled": True, **queue},
    )


def _queued(capacity: str) -> int:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as redis:
        count: Final = redis.zcard(_keys(f"deployment:{capacity}")[1])
    assert isinstance(count, int)
    return count


@pytest.fixture
def capacity() -> str:
    return f"dynamic-queue-{uuid.uuid4()}"


@pytest.mark.asyncio
@pytest.mark.parametrize(("triggers", "held_back"), [([429, 529], True), ([], False)])
async def test_a_provider_429_holds_back_another_workers_request_only_when_429_is_a_trigger(
    upstream: Upstream, capacity: str, triggers: list[int], held_back: bool
) -> None:
    first: Final = _worker(upstream, capacity, trigger_status_codes=triggers)
    second: Final = _worker(upstream, capacity, trigger_status_codes=triggers, queue_timeout=0.5)
    upstream.replies.put(_rate_limited(retry_after_s=30))

    with pytest.raises(litellm.RateLimitError) as rate_limited:
        _ = await first.acompletion(model="group", messages=MESSAGES)
    assert rate_limited.value.status_code == 429

    if held_back:
        with pytest.raises(DynamicQueueTimeoutError):
            _ = await second.acompletion(model="group", messages=MESSAGES)
        assert upstream.chat_requests() == 1
    else:
        served = await second.acompletion(model="group", messages=MESSAGES)
        assert "served" in str(served)
        assert upstream.chat_requests() == 2


@pytest.mark.asyncio
async def test_a_congested_capacity_serves_a_waiting_request_once_its_probe_succeeds(
    upstream: Upstream, capacity: str
) -> None:
    first: Final = _worker(upstream, capacity)
    second: Final = _worker(upstream, capacity, queue_timeout=30)
    upstream.replies.put(_rate_limited(retry_after_s=None))
    with pytest.raises(litellm.RateLimitError):
        _ = await first.acompletion(model="group", messages=MESSAGES)

    served: Final = await second.acompletion(model="group", messages=MESSAGES)

    assert "served" in str(served)
    assert upstream.chat_requests() == 2


@pytest.mark.asyncio
async def test_a_rate_limit_litellm_raises_itself_for_a_full_deployment_does_not_congest_it(
    upstream: Upstream, capacity: str
) -> None:
    first: Final = _worker(upstream, capacity, max_parallel_requests=1)
    second: Final = _worker(upstream, capacity, queue_timeout=0.5)
    released: Final = threading.Event()
    upstream.replies.put(
        Reply(chunks=(chat_reply("chatcmpl-held", "gpt-x", "held", stream=False).body,), gate_before_first=released)
    )
    holder: Final = asyncio.create_task(first.acompletion(model="group", messages=MESSAGES))
    try:
        _ = await asyncio.to_thread(eventually, lambda: upstream.wire.received.qsize(), lambda count: count >= 1)
        with pytest.raises(litellm.RateLimitError):
            _ = await first.acompletion(model="group", messages=MESSAGES)
        released.set()
        assert "held" in str(await holder)

        served: Final = await second.acompletion(model="group", messages=MESSAGES)
    finally:
        released.set()
        _ = holder.cancel()
        _ = await asyncio.gather(holder, return_exceptions=True)

    assert "served" in str(served)


@pytest.mark.asyncio
async def test_a_full_queue_refuses_with_503_queue_full_and_a_cancelled_waiter_leaves_it(
    upstream: Upstream, capacity: str
) -> None:
    first: Final = _worker(upstream, capacity, max_waiting=1)
    waiting_worker: Final = _worker(upstream, capacity, max_waiting=1, queue_timeout=30)
    refused_worker: Final = _worker(upstream, capacity, max_waiting=1, queue_timeout=30)
    upstream.replies.put(_rate_limited(retry_after_s=30))
    with pytest.raises(litellm.RateLimitError):
        _ = await first.acompletion(model="group", messages=MESSAGES)
    waiter: Final = asyncio.create_task(waiting_worker.acompletion(model="group", messages=MESSAGES))
    try:
        _ = await asyncio.to_thread(eventually, lambda: _queued(capacity), lambda count: count == 1)

        with pytest.raises(DynamicQueueFullError) as full:
            _ = await refused_worker.acompletion(model="group", messages=MESSAGES)
    finally:
        _ = waiter.cancel()
        _ = await asyncio.gather(waiter, return_exceptions=True)

    assert (full.value.status_code, full.value.code) == (503, QUEUE_FULL_CODE)
    assert upstream.chat_requests() == 1
    assert await asyncio.to_thread(eventually, lambda: _queued(capacity), lambda count: count == 0) == 0
