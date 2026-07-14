# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for truncation-aware KV eviction (VLLM_KV_EVICT_TRUNC)."""

import pytest
import torch

import vllm.v1.core.conversation_kv_registry as conversation_kv_registry
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool, evict_truncated_prefix_blocks
from vllm.v1.core.conversation_kv_registry import ConversationKVRegistry
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.request import Request, RequestStatus

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


@pytest.fixture
def _force_platform(monkeypatch):
    # Force a platform so VllmConfig builds on test machines without a
    # matching native vLLM build.
    import vllm.platforms as vllm_platforms
    from vllm.platforms.cpu import CpuPlatform

    if vllm_platforms._current_platform is None or not (
        vllm_platforms._current_platform.is_cuda()
        or vllm_platforms._current_platform.is_cpu()
    ):
        monkeypatch.setattr(vllm_platforms, "_current_platform", CpuPlatform())


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    kv_transfer_params: dict | None = None,
    resumable: bool = False,
) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        resumable=resumable,
    )
    request.kv_transfer_params = kv_transfer_params
    return request


def make_manager(num_blocks: int = 11) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def alloc_and_free(manager: KVCacheManager, request: Request) -> None:
    """Allocate the full prompt, caching full blocks, then free the request."""
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(
        request,
        request.num_tokens,
        num_computed_tokens,
        computed_blocks,
    )
    assert blocks is not None
    manager.free(request)


def free_queue_ids(manager: KVCacheManager) -> list[int]:
    return [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]


def test_evict_truncated_prefix_moves_dead_blocks_to_front():
    manager = make_manager()
    # 3 full blocks + 1 partial block.
    prompt = [i for i in range(3) for _ in range(BLOCK_SIZE)] + [3] * 7
    req0 = make_request("0", prompt)
    alloc_and_free(manager, req0)
    assert len(req0.block_hashes) == 3
    # Free order: unallocated blocks, then the request's chain reversed.
    assert free_queue_ids(manager) == [5, 6, 7, 8, 9, 10, 4, 3, 2, 1]

    # Truncated new turn shares only the first block: LCP = 1.
    prev_hashes = req0.block_hashes
    assert manager.evict_truncated_prefix(prev_hashes, 1) == (2, 0, 0)

    # Dead blocks (2, 3) are at the queue front, tail-most first; the shared
    # block (1) is untouched.
    assert free_queue_ids(manager) == [3, 2, 5, 6, 7, 8, 9, 10, 4, 1]
    # Dead hashes are gone from the prefix cache; the shared one remains.
    assert manager.block_pool.get_cached_block(prev_hashes[0], [0]) is not None
    assert manager.block_pool.get_cached_block(prev_hashes[1], [0]) is None
    assert manager.block_pool.get_cached_block(prev_hashes[2], [0]) is None

    # The next allocation consumes the dead blocks first.
    new_blocks = manager.block_pool.get_new_blocks(2)
    assert [b.block_id for b in new_blocks] == [3, 2]

    # A repeated call is a no-op (all misses).
    assert manager.evict_truncated_prefix(prev_hashes, 1) == (0, 0, 2)


def test_evict_truncated_prefix_skips_active_blocks():
    manager = make_manager()
    prompt = [i for i in range(3) for _ in range(BLOCK_SIZE)]
    req0 = make_request("0", prompt)
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(req0)
    blocks = manager.allocate_slots(
        req0, req0.num_tokens, num_computed_tokens, computed_blocks
    )
    assert blocks is not None

    # Request still holds its blocks: everything must be skipped.
    assert manager.evict_truncated_prefix(req0.block_hashes, 0) == (0, 3, 0)
    assert manager.block_pool.get_cached_block(req0.block_hashes[2], [0]) is not None

    manager.free(req0)
    assert manager.evict_truncated_prefix(req0.block_hashes, 0) == (3, 0, 0)


def test_evict_truncated_prefix_blocks_multi_group_block_sizes():
    """Group conversion: hash_block_size=16 with a block_size=32 group.

    Mirrors how the CPU offload pool stamps BlockHashWithGroupId keys
    directly onto its blocks.
    """
    hash_block_size = 16
    groups = [
        KVCacheGroupSpec(
            ["layer1"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["layer2"],
            FullAttentionSpec(
                block_size=32, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
    ]
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)

    prev_hashes = [BlockHash(bytes([i]) * 32) for i in range(4)]
    # Group 0 keys at block_size 16; group 1 keys are concatenated pairs.
    keys = [make_block_hash_with_group_id(h, 0) for h in prev_hashes]
    keys += [
        make_block_hash_with_group_id(BlockHash(prev_hashes[0] + prev_hashes[1]), 1),
        make_block_hash_with_group_id(BlockHash(prev_hashes[2] + prev_hashes[3]), 1),
    ]
    blocks = pool.get_new_blocks(len(keys))
    for block, key in zip(blocks, keys):
        block._block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)
    pool.free_blocks(blocks)

    # LCP of 2 hash blocks (32 tokens): group 0 keeps 2 blocks, group 1
    # keeps 1 block.
    assert evict_truncated_prefix_blocks(
        pool, groups, hash_block_size, prev_hashes, lcp_blocks=2
    ) == (3, 0, 0)
    assert pool.cached_block_hash_to_block.get_one_block(keys[0]) is not None
    assert pool.cached_block_hash_to_block.get_one_block(keys[1]) is not None
    assert pool.cached_block_hash_to_block.get_one_block(keys[2]) is None
    assert pool.cached_block_hash_to_block.get_one_block(keys[3]) is None
    assert pool.cached_block_hash_to_block.get_one_block(keys[4]) is not None
    assert pool.cached_block_hash_to_block.get_one_block(keys[5]) is None


def test_conversation_kv_registry_lru_and_ttl(monkeypatch):
    fake_time = [0.0]
    monkeypatch.setattr(
        conversation_kv_registry.time, "monotonic", lambda: fake_time[0]
    )
    registry = ConversationKVRegistry(max_entries=2, ttl_sec=10)
    h = [BlockHash(bytes([i]) * 32) for i in range(3)]

    registry.record("c1", [h[0]])
    registry.record("c2", [h[1]])
    assert registry.get("c1") == (h[0],)

    # c2 is now the LRU entry and gets dropped at capacity.
    registry.record("c3", [h[2]])
    assert registry.get("c2") is None
    assert registry.get("c1") == (h[0],)
    assert registry.get("c3") == (h[2],)

    # TTL expiry.
    fake_time[0] = 11.0
    assert registry.get("c1") is None
    assert len(registry) == 1


def test_scheduler_truncation_evict_end_to_end(monkeypatch, _force_platform):
    monkeypatch.setenv("VLLM_KV_EVICT_TRUNC", "1")
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    assert scheduler.truncation_evict_registry is not None

    # Turn 1: 4 full blocks, conversation c1.
    turn1_prompt = [i for i in range(4) for _ in range(BLOCK_SIZE)]
    req1 = make_request("t1", turn1_prompt, {"conversation_id": "c1"})
    scheduler.add_request(req1)
    scheduler.schedule()
    prev_hashes = list(req1.block_hashes)
    assert len(prev_hashes) == 4
    scheduler.finish_requests("t1", RequestStatus.FINISHED_ABORTED)
    assert scheduler.truncation_evict_registry.get("c1") == tuple(prev_hashes)

    block_pool = scheduler.kv_cache_manager.block_pool
    for h in prev_hashes:
        assert block_pool.get_cached_block(h, [0]) is not None

    # Turn 2: truncated; shares only the first block with turn 1.
    turn2_prompt = turn1_prompt[:BLOCK_SIZE] + [99] * (BLOCK_SIZE * 3)
    req2 = make_request(
        "t2", turn2_prompt, {"conversation_id": "c1", "truncated": True}
    )
    scheduler.add_request(req2)

    # Blocks beyond the shared prefix are evicted; the shared block remains.
    assert block_pool.get_cached_block(prev_hashes[0], [0]) is not None
    for h in prev_hashes[1:]:
        assert block_pool.get_cached_block(h, [0]) is None
    # Dead blocks sit at the free queue front (tail-most first).
    front_ids = [
        b.block_id for b in block_pool.free_block_queue.get_all_free_blocks()[:3]
    ]
    assert front_ids == [4, 3, 2]

    # A non-truncated turn of another conversation must be unaffected.
    req3 = make_request("t3", turn1_prompt, {"conversation_id": "c2"})
    scheduler.add_request(req3)
    assert block_pool.get_cached_block(prev_hashes[0], [0]) is not None


def test_evict_free_cached_blocks_evicts_duplicate_blocks():
    """Two idle blocks cached under the same hash are both evicted."""
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=BLOCK_SIZE)
    key = make_block_hash_with_group_id(BlockHash(b"\x01" * 32), 0)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        block._block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)
    pool.free_blocks(blocks)

    assert pool.evict_free_cached_blocks([key]) == (2, 0, 0)
    assert pool.cached_block_hash_to_block.get_one_block(key) is None


def test_scheduler_truncation_evict_skips_streaming_requests(
    monkeypatch, _force_platform
):
    """Streaming-input (resumable) requests carry only a partial prompt on
    arrival, so the LCP would be underestimated; eviction must not run."""
    monkeypatch.setenv("VLLM_KV_EVICT_TRUNC", "1")
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    assert scheduler.truncation_evict_registry is not None

    turn1_prompt = [i for i in range(4) for _ in range(BLOCK_SIZE)]
    req1 = make_request("t1", turn1_prompt, {"conversation_id": "c1"})
    scheduler.add_request(req1)
    scheduler.schedule()
    prev_hashes = list(req1.block_hashes)
    scheduler.finish_requests("t1", RequestStatus.FINISHED_ABORTED)

    # Truncated turn 2 arrives as the first chunk of a streaming session,
    # sharing only the first block so far.
    turn2_chunk = turn1_prompt[:BLOCK_SIZE] + [99] * BLOCK_SIZE
    req2 = make_request(
        "t2",
        turn2_chunk,
        {"conversation_id": "c1", "truncated": True},
        resumable=True,
    )
    scheduler.add_request(req2)

    # Nothing was evicted: later chunks may still hit the previous chain.
    block_pool = scheduler.kv_cache_manager.block_pool
    for h in prev_hashes:
        assert block_pool.get_cached_block(h, [0]) is not None


def test_scheduler_truncation_evict_gpu_and_cpu_full_chain(
    monkeypatch, _force_platform
):
    """Plan verification #2: real Scheduler + SimpleCPUOffloadConnector
    (eager), two-turn conversation with a truncation flag; the dead suffix
    is dropped from both the GPU prefix cache and the CPU offload pool."""
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
        SimpleCPUOffloadConnector,
    )
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
    from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata

    monkeypatch.setenv("VLLM_KV_EVICT_TRUNC", "1")
    num_gpu_blocks = 64
    scheduler = create_scheduler(
        enable_prefix_caching=True, block_size=BLOCK_SIZE, num_blocks=num_gpu_blocks
    )
    assert scheduler.truncation_evict_registry is not None

    # Attach a real eager connector with a small CPU pool. The scheduler's
    # kv_cache_config has empty kv_cache_tensors, which _derive_cpu_config
    # needs to size the CPU pool, so mirror it with tensors filled in.
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    bytes_per_block = spec.page_size_bytes
    gpu_kv_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=bytes_per_block * num_gpu_blocks, shared_by=["layer"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    scheduler.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": bytes_per_block * 8},
    )
    connector = SimpleCPUOffloadConnector(
        scheduler.vllm_config, KVConnectorRole.SCHEDULER, gpu_kv_config
    )
    scheduler.connector = connector
    connector.bind_gpu_block_pool(scheduler.kv_cache_manager.block_pool)
    offload_manager = connector.scheduler_manager
    assert offload_manager is not None and not offload_manager._lazy_mode

    # Turn 1: 4 full blocks. Eager mode stores blocks only once their KV is
    # confirmed computed, so run the prefill step, report a model output,
    # and let the next (decode) step emit the store event.
    turn1_prompt = [i for i in range(4) for _ in range(BLOCK_SIZE)]
    req1 = make_request("t1", turn1_prompt, {"conversation_id": "c1"})
    scheduler.add_request(req1)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=["t1"],
            req_id_to_index={"t1": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    decode_output = scheduler.schedule()
    meta = decode_output.kv_connector_metadata
    assert meta is not None and meta.store_event >= 0
    connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                completed_store_events={
                    meta.store_event: offload_manager._expected_worker_count
                }
            )
        )
    )
    prev_hashes = list(req1.block_hashes)
    scheduler.finish_requests("t1", RequestStatus.FINISHED_ABORTED)

    cpu_pool = offload_manager.cpu_block_pool
    cpu_keys = [make_block_hash_with_group_id(h, 0) for h in prev_hashes]
    for key in cpu_keys:
        assert cpu_pool.cached_block_hash_to_block.get_one_block(key) is not None

    # Turn 2: truncated, shares only the first block.
    turn2_prompt = turn1_prompt[:BLOCK_SIZE] + [99] * (BLOCK_SIZE * 3)
    req2 = make_request(
        "t2", turn2_prompt, {"conversation_id": "c1", "truncated": True}
    )
    scheduler.add_request(req2)

    # GPU: dead suffix gone, shared prefix kept.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.get_cached_block(prev_hashes[0], [0]) is not None
    for h in prev_hashes[1:]:
        assert block_pool.get_cached_block(h, [0]) is None
    # CPU: same.
    assert cpu_pool.cached_block_hash_to_block.get_one_block(cpu_keys[0]) is not None
    for key in cpu_keys[1:]:
        assert cpu_pool.cached_block_hash_to_block.get_one_block(key) is None
