# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.core.private_swa_pool import (
    PrivateSWAAllocationPool,
    PrivateSWAConfig,
    PrivateSWAIdCodec,
    compute_prefix_rollback_start,
    compute_private_swa_prefill_workspace_blocks,
    detect_new_prefix_hit_length,
)


def test_compact_sizing_and_allocation_lifecycle():
    config = PrivateSWAConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=4
    )
    assert config.capacity_tokens == 132
    assert config.blocks_per_allocation == 5
    assert config.num_allocations == 8
    assert config.num_blocks == 41

    pool = PrivateSWAAllocationPool(config)
    allocation = pool.reserve("request-1")
    assert allocation is not None
    assert pool.reserve("request-1") == allocation
    assert list(pool.allocation_block_ids(allocation)) == list(
        range(1 + allocation * 5, 1 + allocation * 5 + 5)
    )
    assert pool.slot_for_position(allocation, 0) == (1 + allocation * 5) * 32
    assert pool.slot_for_position(allocation, 128) == (1 + allocation * 5 + 4) * 32
    assert pool.release("request-1")
    assert not pool.release("request-1")


def test_retained_allocation_survives_request_id_reuse():
    config = PrivateSWAConfig(
        block_size=128, window_size=128, in_flight_tokens=1,
        max_num_seqs=1,
    )
    pool = PrivateSWAAllocationPool(config)
    old = pool.reserve("request-1")
    assert old is not None
    pool.acquire_ref(old)
    assert pool.release("request-1")

    new = pool.reserve("request-1")
    assert new is not None and new != old
    assert pool.retained_allocations == 1
    assert pool.release_ref(old)
    assert pool.retained_allocations == 0
    assert pool.ref_count(old) == 0


def test_prefill_workspace_covers_block_aligned_rollback_batch():
    max_num_batched_tokens = 2048
    max_num_seqs = 32
    window_size = 128
    block_size = 32
    workspace_blocks = compute_private_swa_prefill_workspace_blocks(
        max_num_batched_tokens,
        max_num_seqs,
        window_size,
        block_size,
    )

    # This valid batch exposes the old W-1 formula: every rollback request
    # starts a separately aligned workspace range and the query lengths sum to
    # max_num_batched_tokens.
    query_lens = [2017] + [1] * (max_num_seqs - 1)
    required_blocks = sum(
        (window_size + query_len + block_size - 1) // block_size
        for query_len in query_lens
    )
    old_workspace_blocks = (
        max_num_batched_tokens
        + max_num_seqs * (window_size - 1 + block_size - 1)
        + block_size
        - 1
    ) // block_size

    assert sum(query_lens) == max_num_batched_tokens
    assert old_workspace_blocks == 222
    assert required_blocks == workspace_blocks == 223


def test_prefill_workspace_size_rejects_invalid_inputs():
    invalid_args = (
        (-1, 1, 128, 32),
        (1, 0, 128, 32),
        (1, 1, 0, 32),
        (1, 1, 128, 0),
    )
    for args in invalid_args:
        try:
            compute_private_swa_prefill_workspace_blocks(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid workspace sizing input was accepted")


def test_common_scheduler_prefix_hit_detection():
    # A local or synchronous external hit leaves H cached while only P-H new
    # tokens are scheduled.
    assert detect_new_prefix_hit_length(0, 260, 4) == 256

    # Async KV loading may expose H without scheduling a model token, and some
    # scheduler versions set H before entering _update_after_schedule.
    assert detect_new_prefix_hit_length(0, 256, 0) == 256
    assert detect_new_prefix_hit_length(256, 256, 0, True) == 256

    # Ordinary chunk/decode work starts from an already computed prefix and
    # must not trigger one-shot SWA rollback.
    assert detect_new_prefix_hit_length(256, 260, 4) is None
    assert detect_new_prefix_hit_length(0, 4, 4) is None


def test_common_scheduler_prefix_hit_detection_rejects_negative_counts():
    for args in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
        try:
            detect_new_prefix_hit_length(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("negative scheduler token count was accepted")


def test_prefix_rollback_start_is_exact_and_bounded():
    assert compute_prefix_rollback_start(512, 128) == 384
    assert compute_prefix_rollback_start(100, 128) == 0
    assert compute_prefix_rollback_start(513, 0) == 513


def test_compact_ring_wraps_at_absolute_block_boundary():
    config = PrivateSWAConfig(
        block_size=32, window_size=128, in_flight_tokens=5, max_num_seqs=1
    )
    pool = PrivateSWAAllocationPool(config)
    allocation = pool.reserve("ring")
    assert allocation is not None
    blocks = list(pool.allocation_block_ids(allocation))
    # 5 physical blocks hold 160 token slots; absolute positions alias by
    # physical block while preserving the intra-block offset.
    assert pool.slot_for_position(allocation, 0) == blocks[0] * 32
    assert pool.slot_for_position(allocation, 159) == blocks[4] * 32 + 31
    assert pool.slot_for_position(allocation, 160) == blocks[0] * 32
    assert pool.slot_for_position(allocation, 161) == blocks[0] * 32 + 1


def test_pool_exhaustion_and_reuse_after_release():
    config = PrivateSWAConfig(
        block_size=32, window_size=32, in_flight_tokens=1,
        max_num_seqs=2, delayed_free_slots=0,
    )
    pool = PrivateSWAAllocationPool(config)
    first = pool.reserve("first")
    second = pool.reserve("second")
    assert first is not None and second is not None and first != second
    assert pool.reserve("third") is None
    assert pool.release("first")
    assert pool.reserve("third") == first


def test_pool_rejects_invalid_inputs():
    for kwargs in (
        dict(block_size=0, window_size=32, in_flight_tokens=1, max_num_seqs=1),
        dict(block_size=32, window_size=0, in_flight_tokens=1, max_num_seqs=1),
        dict(block_size=32, window_size=32, in_flight_tokens=0, max_num_seqs=1),
        dict(block_size=32, window_size=32, in_flight_tokens=1, max_num_seqs=0),
    ):
        try:
            PrivateSWAAllocationPool(PrivateSWAConfig(**kwargs))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid private SWA sizing was accepted")

    config = PrivateSWAConfig(
        block_size=32, window_size=32, in_flight_tokens=1, max_num_seqs=1
    )
    pool = PrivateSWAAllocationPool(config)
    allocation = pool.reserve("invalid")
    assert allocation is not None
    try:
        pool.slot_for_position(allocation, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative absolute position was accepted")


def test_pool_aware_id_codec_keeps_null_and_private_sentinel_distinct():
    codec = PrivateSWAIdCodec(shared_num_blocks=10, private_num_blocks=8)
    assert codec.decode(0) is None
    private_sentinel = codec.encode("private", 0)
    assert private_sentinel != 0
    assert codec.decode(private_sentinel) == ("private", 0)
    assert codec.decode_for_kernel(private_sentinel, "private") == 0
    assert codec.encode("shared", 0) == 1
    assert codec.decode_for_kernel(codec.encode("shared", 3), "shared") == 3

    try:
        codec.decode_for_kernel(codec.encode("shared", 1), "private")
    except ValueError:
        pass
    else:
        raise AssertionError("pool mismatch was not rejected")

    try:
        codec.decode(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative encoded id was accepted")


def test_prefix_rollback_short_and_exact_boundaries():
    expected = {
        0: 0,
        31: 0,
        127: 0,
        128: 0,
        129: 1,
        255: 127,
        256: 128,
        384: 256,
        260: 132,
    }
    for hit_length, rollback_start in expected.items():
        assert compute_prefix_rollback_start(hit_length, 128) == rollback_start


def test_prefix_rollback_rejects_invalid_parameters():
    for args in ((-1, 128), (128, -1)):
        try:
            compute_prefix_rollback_start(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rollback parameters were accepted")


def test_codec_accepts_highest_delayed_free_allocation():
    config = PrivateSWAConfig(
        block_size=32,
        window_size=128,
        in_flight_tokens=5,
        max_num_seqs=4,
    )
    highest_allocation = config.num_allocations - 1
    highest_block = list(
        PrivateSWAAllocationPool(config).allocation_block_ids(
            highest_allocation
        )
    )[-1]
    codec = PrivateSWAIdCodec(
        shared_num_blocks=0,
        private_num_blocks=config.num_blocks,
    )
    encoded = codec.encode("private", highest_block)
    assert codec.decode_for_kernel(encoded, "private") == highest_block
