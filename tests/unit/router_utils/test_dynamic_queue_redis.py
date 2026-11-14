from collections.abc import Awaitable, Callable, Sequence
from typing import Final

import pytest
from redis.crc import key_slot
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.router_utils.dynamic_queue import CoordinationUnavailable, DynamicQueue, WaitRequest
from litellm.router_utils.dynamic_queue_redis import RedisQueueBackend, _keys, rank_key


def test_rank_keys_sort_byte_wise_in_numeric_order_for_any_integer() -> None:
    priorities: Final = (7, -(10**40), 0, 1001, -1, 10**40, -999, 9, -1001, 10, -10, 999, 1, -9, 10**40 + 1)

    assert sorted(priorities, key=lambda priority: rank_key(priority).encode()) == sorted(priorities)


@pytest.mark.parametrize("capacity", ["deployment-1", "a}b", "}starts-with-brace", "{already-tagged}", ""])
def test_every_key_of_a_capacity_lands_on_one_cluster_slot(capacity: str) -> None:
    slots: Final = {key_slot(key.encode()) for key in _keys(capacity)}

    assert len(slots) == 1


class UnreachableRedis:
    def async_register_script(self, script: str) -> Callable[..., Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[str | int]) -> object:
            raise RedisConnectionError("connection refused")

        return run


async def test_an_unreachable_redis_rejects_the_wait_instead_of_admitting_locally() -> None:
    queue: Final = DynamicQueue(RedisQueueBackend(UnreachableRedis()))

    result: Final = await queue.admit(WaitRequest(capacity="deployment-1", waiter_id="a", priority=0), 30.0)

    assert isinstance(result, CoordinationUnavailable)
    assert "connection refused" in result.reason
