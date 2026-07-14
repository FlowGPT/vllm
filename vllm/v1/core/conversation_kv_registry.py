# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-conversation record of the last turn's KV block hashes.

Used by truncation-aware eviction (VLLM_KV_EVICT_TRUNC): when a multi-turn
chat client reports that a turn was sliding-window truncated, the scheduler
compares the new prompt's block-hash chain against the conversation's
previous chain recorded here to find blocks that can never be hit again.
"""

import time
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash


class ConversationKVRegistry:
    """LRU + TTL map: conversation_id -> block hashes of the last turn.

    Entries hold reference copies of the request's ``block_hashes`` list
    (cheap; the bytes objects are shared). Not thread-safe; only accessed
    from the scheduler loop.
    """

    def __init__(self, max_entries: int, ttl_sec: float):
        self.max_entries = max_entries
        self.ttl_sec = ttl_sec
        # conversation_id -> (record_time, block_hashes)
        self._entries: OrderedDict[str, tuple[float, tuple["BlockHash", ...]]] = (
            OrderedDict()
        )

    def record(self, conversation_id: str, block_hashes: list["BlockHash"]) -> None:
        self._entries.pop(conversation_id, None)
        self._entries[conversation_id] = (time.monotonic(), tuple(block_hashes))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def get(self, conversation_id: str) -> tuple["BlockHash", ...] | None:
        entry = self._entries.get(conversation_id)
        if entry is None:
            return None
        record_time, block_hashes = entry
        if time.monotonic() - record_time > self.ttl_sec:
            del self._entries[conversation_id]
            return None
        self._entries.move_to_end(conversation_id)
        return block_hashes

    def __len__(self) -> int:
        return len(self._entries)
