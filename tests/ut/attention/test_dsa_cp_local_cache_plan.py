# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import (
    DSACP_LOCAL_CACHE_UNIT_SIZE,
    AscendDSACPImpl,
    DSACPStateBroadcastPlan,
    _select_indexer_hidden_states_full,
    build_dsa_cp_hidden_input_plan,
    build_dsa_cp_local_cache_plan,
    get_dsa_cp_all_rank_token_counts,
    build_dsa_cp_local_compressed_range,
    build_dsa_cp_local_compressor_slot_plan,
    build_dsa_cp_state_broadcast_plan,
    build_dsa_cp_swa_window_plan,
)
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy, concatenate_rope_slices


def test_local_cache_cp_concatenates_rope_proxy_ranges():
    cos = torch.arange(24).reshape(6, 4)
    sin = cos + 100
    rope_data = RopeDataProxy({"config": {"default": (cos, sin)}})

    local_cos, local_sin = concatenate_rope_slices(rope_data, ((0, 2), (4, 6)))

    actual_cos, actual_sin = local_cos._data["config"]["default"]
    assert local_cos.idx == 0
    assert local_sin.idx == 1
    assert torch.equal(actual_cos, torch.cat((cos[0:2], cos[4:6])))
    assert torch.equal(actual_sin, torch.cat((sin[0:2], sin[4:6])))


def test_legacy_cp_indexer_uses_unpadded_hidden_states_cache():
    hidden_states_full = torch.arange(10)
    hidden_states_cache = hidden_states_full[:9]

    selected = _select_indexer_hidden_states_full(
        hidden_states_full,
        hidden_states_cache,
        use_local_cache_prefill=False,
    )

    assert selected is hidden_states_cache
    assert selected.shape[0] == 9


def test_local_cache_cp_preserves_full_hidden_states_semantics():
    hidden_states_cache = torch.arange(9)

    selected = _select_indexer_hidden_states_full(
        None,
        hidden_states_cache,
        use_local_cache_prefill=True,
    )

    assert selected is None


def _plans(num_input_tokens: int, cp_size: int):
    return [
        build_dsa_cp_local_cache_plan(num_input_tokens, cp_size, rank)
        for rank in range(cp_size)
    ]


def test_cp64_9k_single_request_uses_128_token_units():
    plans = _plans(num_input_tokens=9 * 1024, cp_size=64)

    assert [plan.local_num_tokens for plan in plans[:8]] == [256] * 8
    assert [plan.local_num_tokens for plan in plans[8:]] == [128] * 56
    assert plans[0].local_start == 0
    assert plans[7].local_start == 7 * 256
    assert plans[8].local_start == 2048
    assert plans[-1].local_end == 9 * 1024
    assert {plan.num_tokens_pad for plan in plans} == {9 * 1024}


def test_padding_is_applied_per_request_before_flatten_cp_split():
    query_start_loc = [0, 9 * 1024, 9 * 1024 + 1]
    plans = [
        build_dsa_cp_local_cache_plan(
            num_input_tokens=9 * 1024 + 1,
            cp_size=64,
            cp_rank=rank,
            query_start_loc=query_start_loc,
        )
        for rank in range(64)
    ]

    assert {plan.num_tokens_pad for plan in plans} == {9 * 1024 + DSACP_LOCAL_CACHE_UNIT_SIZE}
    assert sum(plan.local_num_tokens for plan in plans) == 9 * 1024 + 1
    assert plans[0].rank_valid_ranges[0] == ((0, 256), (9 * 1024, 9 * 1024))
    assert plans[-1].rank_valid_ranges[-1] == ((9 * 1024, 9 * 1024), (9 * 1024, 9 * 1024 + 1))


def test_short_batch_assigns_padding_units_but_only_counts_valid_tokens():
    plans = _plans(num_input_tokens=1, cp_size=64)

    assert [plan.local_num_tokens for plan in plans] == [1] + [0] * 63
    assert {plan.num_tokens_pad for plan in plans} == {DSACP_LOCAL_CACHE_UNIT_SIZE}
    assert plans[0].local_valid_ranges == ((0, 1),)
    assert all(plan.local_valid_ranges == () for plan in plans[1:])


@pytest.mark.parametrize(
    ("num_input_tokens", "cp_size", "cp_rank"),
    [
        (128, 0, 0),
        (128, 2, -1),
        (128, 2, 2),
    ],
)
def test_local_cache_plan_rejects_invalid_cp_args(num_input_tokens, cp_size, cp_rank):
    with pytest.raises(ValueError):
        build_dsa_cp_local_cache_plan(num_input_tokens, cp_size, cp_rank)


def test_swa_window_plan_uses_previous_128_tokens_within_request():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 9 * 1024],
        num_actual_tokens=9 * 1024,
    )

    assert window_plan.valid_ranges == [(2048, 2176)]
    assert window_plan.halo_ranges == [(1920, 2048)]
    assert window_plan.input_ranges == [(1920, 2176)]
    assert [slots.shape[0] for slots in window_plan.all_rank_slot_mappings[:9]] == [256] * 8 + [128]


def test_hidden_input_plan_splits_swa_halo_and_local_sources():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 9 * 1024],
        num_actual_tokens=9 * 1024,
    )

    hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=window_plan.input_ranges,
        local_cache_plan=plan,
        num_actual_tokens=9 * 1024,
    )

    assert hidden_plan.input_ranges == [(1920, 2176)]
    assert hidden_plan.halo_source_ranges == [(1920, 2048)]
    assert hidden_plan.halo_output_ranges == [(0, 128)]
    assert hidden_plan.local_source_ranges == [(2048, 2176)]
    assert hidden_plan.local_read_ranges == [(0, 128)]
    assert hidden_plan.local_output_ranges == [(128, 256)]
    assert hidden_plan.num_input_tokens == 256


def test_swa_window_plan_does_not_borrow_halo_across_requests():
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=384, cp_size=2, cp_rank=1, query_start_loc=[0, 192, 384]
    )
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 192, 384],
        num_actual_tokens=384,
    )

    assert window_plan.valid_ranges == [(192, 384)]
    assert window_plan.halo_ranges == [(192, 192)]
    assert window_plan.input_ranges == [(192, 384)]


def test_swa_window_plan_clips_padding_tail():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=129, cp_size=4, cp_rank=2)
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 129],
        num_actual_tokens=129,
    )

    assert window_plan.valid_ranges == []
    assert window_plan.halo_ranges == []
    assert window_plan.input_ranges == []


def test_c128_local_compressed_range_uses_owner_token_slice():
    input_positions = torch.arange(9 * 1024)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)

    compressed_start, compressed_end = build_dsa_cp_local_compressed_range(
        input_positions=input_positions,
        compress_ratio=128,
        local_cache_plan=plan,
        num_actual_tokens=9 * 1024,
    )

    assert (compressed_start, compressed_end) == (16, 17)


def test_local_compressed_range_uses_per_request_owner_ranges():
    input_positions = torch.cat([torch.arange(192), torch.arange(192)])
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=384, cp_size=2, cp_rank=1, query_start_loc=[0, 192, 384]
    )

    compressed_start, compressed_end = build_dsa_cp_local_compressed_range(
        input_positions=input_positions,
        compress_ratio=128,
        local_cache_plan=plan,
        num_actual_tokens=384,
    )

    assert (compressed_start, compressed_end) == (1, 2)


def test_local_compressed_range_handles_padding_tail_rank():
    input_positions = torch.arange(129)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=129, cp_size=4, cp_rank=2)

    compressed_start, compressed_end = build_dsa_cp_local_compressed_range(
        input_positions=input_positions,
        compress_ratio=128,
        local_cache_plan=plan,
        num_actual_tokens=129,
    )

    assert (compressed_start, compressed_end) == (1, 1)


def test_c4_local_compressor_slot_plan_marks_overlap_outputs_invalid():
    input_positions = torch.arange(136)
    slot_mapping = torch.arange(1000, 1000 + 34)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=136, cp_size=2, cp_rank=1)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 136],
        num_actual_tokens=136,
        overlap_tokens=4,
    )

    assert compressor_plan.input_ranges == [(124, 136)]
    assert compressor_plan.valid_ranges == [(128, 136)]
    assert compressor_plan.overlap_ranges == [(124, 128)]
    assert compressor_plan.slot_mapping.tolist() == [-1, 1032, 1033]
    assert compressor_plan.valid_output_mask.tolist() == [False, True, True]
    assert compressor_plan.output_indices.tolist() == [31, 32, 33]
    assert compressor_plan.compressed_positions.tolist() == [124, 128, 132]
    assert compressor_plan.input_indices.tolist() == list(range(124, 136))
    assert compressor_plan.input_query_start_loc.tolist() == [0, 12]
    assert compressor_plan.request_indices.tolist() == [0]
    assert compressor_plan.start_pos_offsets.tolist() == [124]
    assert [slots.tolist() for slots in compressor_plan.all_rank_slot_mappings] == [
        list(range(1000, 1032)),
        [1032, 1033],
    ]


def test_local_compressor_slot_plan_does_not_borrow_across_requests():
    query_start_loc = [0, 132, 264]
    input_positions = torch.cat([torch.arange(132), torch.arange(132)])
    slot_mapping = torch.arange(2000, 2000 + 66)
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=264, cp_size=3, cp_rank=1, query_start_loc=query_start_loc
    )

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=264,
        overlap_tokens=4,
    )

    assert compressor_plan.input_ranges == [(132, 260)]
    assert compressor_plan.valid_ranges == [(132, 260)]
    assert compressor_plan.overlap_ranges == [(132, 132)]
    assert compressor_plan.slot_mapping[:2].tolist() == [2033, 2034]
    assert compressor_plan.valid_output_mask[:2].tolist() == [True, True]
    assert compressor_plan.output_indices[:2].tolist() == [33, 34]
    assert compressor_plan.compressed_positions[:2].tolist() == [0, 4]
    assert compressor_plan.input_indices.tolist() == list(range(132, 260))
    assert compressor_plan.input_query_start_loc.tolist() == [0, 128]
    assert compressor_plan.request_indices.tolist() == [1]
    assert compressor_plan.start_pos_offsets.tolist() == [0]


def test_c128_local_compressor_slot_plan_marks_borrowed_rank_tail_invalid():
    input_positions = torch.arange(9 * 1024)
    slot_mapping = torch.arange(3000, 3000 + 72)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=128,
        local_cache_plan=plan,
        query_start_loc=[0, 9 * 1024],
        num_actual_tokens=9 * 1024,
        overlap_tokens=128,
    )

    assert compressor_plan.input_ranges == [(1920, 2176)]
    assert compressor_plan.valid_ranges == [(2048, 2176)]
    assert compressor_plan.overlap_ranges == [(1920, 2048)]
    assert compressor_plan.slot_mapping.tolist() == [-1, 3016]
    assert compressor_plan.valid_output_mask.tolist() == [False, True]
    assert compressor_plan.output_indices.tolist() == [15, 16]
    assert compressor_plan.compressed_positions.tolist() == [1920, 2048]
    assert compressor_plan.input_indices.tolist() == list(range(1920, 2176))
    assert compressor_plan.input_query_start_loc.tolist() == [0, 256]
    assert compressor_plan.request_indices.tolist() == [0]
    assert compressor_plan.start_pos_offsets.tolist() == [1920]


def test_local_compressor_slot_plan_records_all_rank_valid_output_counts():
    input_positions = torch.arange(9 * 1024)
    slot_mapping = torch.arange(3000, 3000 + 72)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=128,
        local_cache_plan=plan,
        query_start_loc=[0, 9 * 1024],
        num_actual_tokens=9 * 1024,
        overlap_tokens=128,
    )

    assert compressor_plan.all_rank_valid_output_counts[:8] == (2,) * 8
    assert compressor_plan.all_rank_valid_output_counts[8:] == (1,) * 56
    assert sum(compressor_plan.all_rank_valid_output_counts) == 72


def test_local_compressor_slot_plan_preserves_2d_slot_mapping_shape():
    input_positions = torch.arange(136)
    slot_mapping = torch.stack([torch.arange(34), torch.arange(100, 134)], dim=-1)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=136, cp_size=2, cp_rank=1)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 136],
        num_actual_tokens=136,
        overlap_tokens=4,
    )

    assert compressor_plan.slot_mapping.shape == (3, 2)
    assert compressor_plan.slot_mapping.tolist() == [[-1, -1], [32, 132], [33, 133]]
    assert compressor_plan.output_indices.tolist() == [31, 32, 33]
    assert compressor_plan.compressed_positions.tolist() == [124, 128, 132]
    assert compressor_plan.input_query_start_loc.tolist() == [0, 12]


def test_state_broadcast_plan_uses_request_tail_owner_rank():
    query_start_loc = [0, 2048, 2176, 9 * 1024]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=9 * 1024, cp_size=64, cp_rank=8, query_start_loc=query_start_loc
    )

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=9 * 1024,
    )

    assert state_plan.source_ranks.tolist() == [7, 8, 63]
    assert state_plan.local_request_indices.tolist() == [1]
    assert state_plan.tail_token_offsets.tolist() == [2047, 2175, 9215]


def test_state_broadcast_plan_handles_per_request_tail_padding():
    query_start_loc = [0, 9 * 1024, 9 * 1024 + 1]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=9 * 1024 + 1, cp_size=64, cp_rank=63, query_start_loc=query_start_loc
    )

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=9 * 1024 + 1,
    )

    assert state_plan.source_ranks.tolist() == [62, 63]
    assert state_plan.local_request_indices.tolist() == [1]
    assert state_plan.tail_token_offsets.tolist() == [9215, 9216]


def test_state_broadcast_plan_marks_empty_requests_invalid():
    query_start_loc = [0, 128, 128, 256]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=256, cp_size=2, cp_rank=0, query_start_loc=query_start_loc
    )

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=256,
    )

    assert state_plan.source_ranks.tolist() == [0, -1, 1]
    assert state_plan.local_request_indices.tolist() == [0]
    assert state_plan.tail_token_offsets.tolist() == [127, -1, 255]


def test_state_broadcast_plan_selects_c128_tail_state_block():
    query_start_loc = [0, 2048, 2176, 9 * 1024]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=9 * 1024, cp_size=64, cp_rank=8, query_start_loc=query_start_loc
    )
    state_block_table = torch.arange(1000, 1000 + 3 * 4, dtype=torch.int32).reshape(3, 4)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=9 * 1024,
        input_positions=torch.arange(9 * 1024),
        state_block_table=state_block_table,
        compress_ratio=128,
        state_block_size=32,
    )

    assert state_plan.source_ranks.tolist() == [7, 8, 63]
    assert state_plan.state_block_indices.tolist() == [0, 0, 2]
    assert state_plan.state_block_ids.tolist() == [1000, 1004, 1010]
    assert state_plan.state_valid_mask.tolist() == [True, True, True]


def test_state_broadcast_plan_selects_c4_tail_state_block():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=136, cp_size=2, cp_rank=1)
    state_block_table = torch.arange(2000, 2000 + 5, dtype=torch.int32).reshape(1, 5)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 136],
        num_actual_tokens=136,
        input_positions=torch.arange(136),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=8,
    )

    assert state_plan.source_ranks.tolist() == [1]
    assert state_plan.local_request_indices.tolist() == [0]
    assert state_plan.tail_token_offsets.tolist() == [135]
    assert state_plan.state_block_indices.tolist() == [4]
    assert state_plan.state_block_ids.tolist() == [2004]
    assert state_plan.state_valid_mask.tolist() == [True]


def test_state_broadcast_plan_uses_request_local_positions_for_state_blocks():
    query_start_loc = [0, 132, 264]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=264, cp_size=3, cp_rank=1, query_start_loc=query_start_loc
    )
    state_block_table = torch.arange(3000, 3000 + 2 * 5, dtype=torch.int32).reshape(2, 5)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=264,
        input_positions=torch.cat([torch.arange(132), torch.arange(132)]),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=8,
    )

    assert state_plan.source_ranks.tolist() == [0, 2]
    assert state_plan.local_request_indices.tolist() == []
    assert state_plan.tail_token_offsets.tolist() == [131, 263]
    assert state_plan.state_block_indices.tolist() == [4, 4]
    assert state_plan.state_block_ids.tolist() == [3004, 3009]
    assert state_plan.state_valid_mask.tolist() == [True, True]


def test_state_broadcast_plan_marks_missing_state_block_invalid():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=256, cp_size=2, cp_rank=0)
    state_block_table = torch.tensor([[4000]], dtype=torch.int32)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 256],
        num_actual_tokens=256,
        input_positions=torch.arange(256),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=8,
    )

    assert state_plan.state_block_indices.tolist() == [-1]
    assert state_plan.state_block_ids.tolist() == [-1]
    assert state_plan.state_valid_mask.tolist() == [False]


def test_state_broadcast_helper_updates_selected_state_blocks(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    state_cache = torch.zeros((5, 2, 1, 3), dtype=torch.float32)
    plan = DSACPStateBroadcastPlan(
        source_ranks=torch.tensor([0, -1, 1], dtype=torch.int32),
        local_request_indices=torch.tensor([], dtype=torch.long),
        tail_token_offsets=torch.tensor([127, -1, 255], dtype=torch.long),
        state_block_ids=torch.tensor([2, 3, 4], dtype=torch.int32),
        state_block_indices=torch.tensor([0, -1, 1], dtype=torch.long),
        state_valid_mask=torch.tensor([True, True, False], dtype=torch.bool),
    )
    calls = []

    def fake_broadcast(buffer, src, group):
        calls.append((src, group))
        buffer.fill_(7.0)

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fake_broadcast,
    )

    impl._broadcast_dsa_cp_state_blocks(state_cache, plan)

    assert calls == [(10, impl.tp_group.device_group)]
    assert torch.all(state_cache[2] == 7.0)
    assert torch.all(state_cache[3] == 0.0)
    assert torch.all(state_cache[4] == 0.0)


def test_state_broadcast_plan_keeps_state_block_ids_on_cpu():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=256, cp_size=2, cp_rank=0)
    state_block_table = torch.tensor([[5000, 5001]], dtype=torch.int32)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 256],
        num_actual_tokens=256,
        input_positions=torch.arange(256),
        state_block_table=state_block_table,
        compress_ratio=128,
        state_block_size=2,
    )

    assert state_plan.state_block_ids.device.type == "cpu"



def test_hidden_halo_gather_uses_allgather(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    plan = build_dsa_cp_local_cache_plan(num_input_tokens=384, cp_size=2, cp_rank=1)
    hidden_states_local = torch.arange(128 * 2, dtype=torch.float32).view(128, 2)
    allgather_calls = []

    def fake_all_gather(output_tensors, input_tensor, group):
        allgather_calls.append((group, input_tensor.clone()))
        output_tensors[0].zero_()
        output_tensors[0][:128].copy_(torch.full((128, 2), -1.0))
        output_tensors[1].copy_(input_tensor)

    def fail_broadcast(*args, **kwargs):
        raise AssertionError("hidden halo should use all_gather, not broadcast")

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.all_gather",
        fake_all_gather,
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fail_broadcast,
    )

    halo_buffers, halo_ranges = impl._gather_dsa_cp_hidden_halos(hidden_states_local, plan, 384)

    assert len(allgather_calls) == 1
    assert halo_ranges == [(128, 256), (256, 384)]
    assert halo_buffers[1].tolist() == hidden_states_local.tolist()

def test_cache_update_allgather_receives_remote_update_from_metadata_slots(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    plan = SimpleNamespace(
        all_rank_valid_output_counts=(2, 0),
        all_rank_slot_mappings=(torch.tensor([5, 6]), torch.empty((0,), dtype=torch.long)),
    )
    cache = torch.zeros((4, 3), dtype=torch.bfloat16)
    calls = []
    allgather_calls = []

    def fake_all_gather(output_tensors, input_tensor, group):
        allgather_calls.append((group, tuple(input_tensor.shape)))
        if len(allgather_calls) == 1:
            output_tensors[0].zero_()
            output_tensors[0][:3] = torch.tensor([2, 2, 3], dtype=output_tensors[0].dtype)
            output_tensors[1].zero_()
        else:
            output_tensors[0].copy_(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=output_tensors[0].dtype))
            output_tensors[1].zero_()

    def fake_scatter(target_cache, update, slot_mapping):
        calls.append((target_cache, update.clone(), slot_mapping.clone()))

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.all_gather",
        fake_all_gather,
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.DeviceOperator.dsa_kv_compress_scatter",
        fake_scatter,
    )

    impl._all_gather_dsa_cp_compressed_cache_updates(cache, None, plan)

    assert [shape for _, shape in allgather_calls] == [(9,), (2, 3)]
    assert len(calls) == 1
    assert calls[0][0] is cache
    assert calls[0][1].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert calls[0][2].tolist() == [5, 6]


def test_cache_update_allgather_uses_local_shape_metadata(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 0
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    plan = SimpleNamespace(
        all_rank_valid_output_counts=(1, 1),
        all_rank_slot_mappings=(torch.tensor([2]), torch.tensor([3])),
    )
    cache = torch.zeros((4, 2), dtype=torch.float32)
    local_update = torch.tensor([[7, 8]], dtype=torch.float32)
    calls = []

    def fake_all_gather(output_tensors, input_tensor, group):
        if input_tensor.shape == (9,):
            output_tensors[0].copy_(input_tensor)
            output_tensors[1].zero_()
            output_tensors[1][:3] = torch.tensor([2, 1, 2], dtype=output_tensors[1].dtype)
        else:
            output_tensors[0].copy_(input_tensor)
            output_tensors[1].copy_(torch.tensor([[3, 4]], dtype=output_tensors[1].dtype))

    def fake_scatter(target_cache, update, slot_mapping):
        calls.append((target_cache, update.clone(), slot_mapping.clone()))

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.all_gather",
        fake_all_gather,
    )
    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.DeviceOperator.dsa_kv_compress_scatter",
        fake_scatter,
    )

    impl._all_gather_dsa_cp_compressed_cache_updates(cache, local_update, plan)

    assert len(calls) == 1
    assert calls[0][1].dtype == torch.float32
    assert calls[0][1].tolist() == [[3, 4]]
    assert calls[0][2].tolist() == [3]


def test_cache_update_allgather_rejects_rank_count_mismatch(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 0
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    plan = SimpleNamespace(
        all_rank_valid_output_counts=(1, 0, 0),
        all_rank_slot_mappings=(
            torch.tensor([0]),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
        ),
    )
    cache = torch.zeros((4, 2), dtype=torch.bfloat16)

    def fail_all_gather(output_tensors, input_tensor, group):
        raise AssertionError("all_gather should not be called for rank-count mismatch")

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.all_gather",
        fail_all_gather,
    )

    with pytest.raises(RuntimeError, match="rank count mismatch"):
        impl._all_gather_dsa_cp_compressed_cache_updates(cache, cache[:1], plan)


def test_hidden_input_plan_preserves_multi_request_boundaries_from_window_plan():
    query_start_loc = [0, 192, 384]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=384, cp_size=2, cp_rank=1, query_start_loc=query_start_loc
    )
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=384,
    )

    hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=window_plan.input_ranges,
        local_cache_plan=plan,
        num_actual_tokens=384,
    )

    assert window_plan.halo_ranges == [(192, 192)]
    assert hidden_plan.input_ranges == [(192, 384)]
    assert hidden_plan.halo_source_ranges == []
    assert hidden_plan.halo_output_ranges == []
    assert hidden_plan.local_source_ranges == [(192, 384)]
    assert hidden_plan.local_read_ranges == [(0, 192)]
    assert hidden_plan.local_output_ranges == [(0, 192)]
    assert hidden_plan.num_input_tokens == 192


def test_swa_window_plan_records_all_rank_valid_token_counts():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)
    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 9 * 1024],
        num_actual_tokens=9 * 1024,
    )

    assert window_plan.all_rank_valid_token_counts[:8] == (256,) * 8
    assert window_plan.all_rank_valid_token_counts[8:] == (128,) * 56


def test_hidden_input_assembly_uses_local_and_halo_sources():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=384, cp_size=2, cp_rank=1)
    hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=[(64, 256)],
        local_cache_plan=plan,
        num_actual_tokens=384,
    )
    hidden_states_local = torch.arange(192, 384).view(192, 1).to(torch.float32)
    rank0_halo = torch.arange(64, 192).view(128, 1).to(torch.float32)
    rank1_halo = torch.arange(256, 384).view(128, 1).to(torch.float32)

    assembled = AscendDSACPImpl._assemble_dsa_cp_hidden_input(
        AscendDSACPImpl,
        hidden_plan,
        hidden_states_local,
        ([rank0_halo, rank1_halo], [(64, 192), (256, 384)]),
    )

    assert assembled.flatten().tolist() == list(range(64, 256))


def test_hidden_input_assembly_rejects_uncovered_halo_without_full_hidden():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=384, cp_size=2, cp_rank=1)
    hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=[(64, 256)],
        local_cache_plan=plan,
        num_actual_tokens=384,
    )
    hidden_states_local = torch.arange(192, 384).view(192, 1).to(torch.float32)
    rank1_halo = torch.arange(256, 384).view(128, 1).to(torch.float32)

    with pytest.raises(RuntimeError, match="hidden halo range"):
        AscendDSACPImpl._assemble_dsa_cp_hidden_input(
            AscendDSACPImpl,
            hidden_plan,
            hidden_states_local,
            ([rank1_halo], [(256, 384)]),
        )


def test_all_rank_token_counts_support_variable_128_unit_split():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=9 * 1024, cp_size=64, cp_rank=8)

    assert get_dsa_cp_all_rank_token_counts(plan) == (256,) * 8 + (128,) * 56


def test_c4_chunk_prefill_compressor_plan_records_prefix_hidden():
    # Current chunk starts at absolute position 2050. The first C4 output owned
    # by this chunk is position 2051 and needs hidden positions [2048, 2052),
    # so two hidden tokens must come from the previous chunk tail cache.
    input_positions = torch.arange(2050, 2058)
    slot_mapping = torch.arange(4000, 4002)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=8, cp_size=1, cp_rank=0)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 8],
        num_actual_tokens=8,
        overlap_tokens=4,
    )

    assert compressor_plan.input_ranges == [(0, 8)]
    assert compressor_plan.prefix_lengths.tolist() == [2]
    assert compressor_plan.current_start_positions.tolist() == [2050]
    assert compressor_plan.has_prefix_hidden is True
    assert compressor_plan.start_pos_offsets.tolist() == [-2]
    assert compressor_plan.input_query_start_loc.tolist() == [0, 10]
    assert compressor_plan.slot_mapping.tolist() == [4000, 4001]
    assert compressor_plan.valid_output_mask.tolist() == [True, True]
    assert compressor_plan.compressed_positions.tolist() == [2048, 2052]


def test_c128_chunk_prefill_compressor_plan_records_prefix_hidden():
    # Current chunk starts exactly at a C128 boundary. The first owner output is
    # at 2175 and needs only current chunk hidden [2048, 2176).
    input_positions = torch.arange(2048, 2176)
    slot_mapping = torch.arange(5000, 5001)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=128, cp_size=1, cp_rank=0)

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=128,
        local_cache_plan=plan,
        query_start_loc=[0, 128],
        num_actual_tokens=128,
        overlap_tokens=128,
    )

    assert compressor_plan.input_ranges == [(0, 128)]
    assert compressor_plan.prefix_lengths.tolist() == [0]
    assert compressor_plan.has_prefix_hidden is False
    assert compressor_plan.start_pos_offsets.tolist() == [0]
    assert compressor_plan.input_query_start_loc.tolist() == [0, 128]
    assert compressor_plan.slot_mapping.tolist() == [5000]
    assert compressor_plan.valid_output_mask.tolist() == [True]
    assert compressor_plan.compressed_positions.tolist() == [2048]


def test_compressor_hidden_input_assembly_uses_previous_chunk_tail_cache():
    input_positions = torch.arange(2050, 2058)
    slot_mapping = torch.arange(4000, 4002)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=8, cp_size=1, cp_rank=0)
    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 8],
        num_actual_tokens=8,
        overlap_tokens=4,
    )
    hidden_input_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=compressor_plan.input_ranges,
        local_cache_plan=plan,
        num_actual_tokens=8,
    )
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl._dsa_cp_hidden_tail_cache = {
        "model.layers.0.self_attn": {"idx:0": (2050, torch.tensor([[100.0], [101.0]]))}
    }
    hidden_states_local = torch.arange(10, 18).view(8, 1).to(torch.float32)

    assembled = impl._assemble_dsa_cp_compressor_hidden_input(
        "model.layers.0.self_attn",
        compressor_plan,
        hidden_input_plan,
        hidden_states_local,
        None,
    )

    assert assembled.flatten().tolist() == [100.0, 101.0] + list(range(10, 18))

def test_compressor_hidden_input_assembly_uses_request_id_tail_cache_key():
    input_positions = torch.arange(2050, 2058)
    slot_mapping = torch.arange(4000, 4002)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=8, cp_size=1, cp_rank=0)
    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 8],
        num_actual_tokens=8,
        overlap_tokens=4,
        request_ids=["req-a"],
    )
    hidden_input_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=compressor_plan.input_ranges,
        local_cache_plan=plan,
        num_actual_tokens=8,
    )
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl._dsa_cp_hidden_tail_cache = {
        "model.layers.0.self_attn": {"req-a": (2050, torch.tensor([[100.0], [101.0]]))}
    }
    hidden_states_local = torch.arange(10, 18).view(8, 1).to(torch.float32)

    assembled = impl._assemble_dsa_cp_compressor_hidden_input(
        "model.layers.0.self_attn",
        compressor_plan,
        hidden_input_plan,
        hidden_states_local,
        None,
    )

    assert assembled.flatten().tolist() == [100.0, 101.0] + list(range(10, 18))


def test_compressor_hidden_input_assembly_requires_matching_tail_cache():
    input_positions = torch.arange(2050, 2058)
    slot_mapping = torch.arange(4000, 4002)
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=8, cp_size=1, cp_rank=0)
    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, 8],
        num_actual_tokens=8,
        overlap_tokens=4,
    )
    hidden_input_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=compressor_plan.input_ranges,
        local_cache_plan=plan,
        num_actual_tokens=8,
    )
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl._dsa_cp_hidden_tail_cache = {}

    with pytest.raises(RuntimeError, match="previous hidden tail cache"):
        impl._assemble_dsa_cp_compressor_hidden_input(
            "model.layers.0.self_attn",
            compressor_plan,
            hidden_input_plan,
            torch.arange(10, 18).view(8, 1).to(torch.float32),
            None,
        )
