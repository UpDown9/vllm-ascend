# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.dsa_cp import (
    DSACP_LOCAL_CACHE_UNIT_SIZE,
    AscendDSACPImpl,
    AscendDSACPMetadataBuilder,
    DSACPStateBroadcastPlan,
    _select_indexer_hidden_states_full,
    build_dsa_cp_hidden_input_plan,
    build_dsa_cp_local_cache_plan,
    build_dsa_cp_local_compressed_range,
    build_dsa_cp_local_compressor_slot_plan,
    build_dsa_cp_state_broadcast_plan,
    build_dsa_cp_swa_window_plan,
    compact_dsa_cp_sample_tokens,
    find_dsa_cp_local_cache_plan,
    get_dsa_cp_all_rank_token_counts,
    get_dsa_cp_state_boundary_granularity,
    materialize_dsa_cp_compressor_slot_plan,
    should_precompute_dsa_cp_compressor_metadata,
    restore_dsa_cp_global_tokens,
    select_dsa_cp_local_tokens,
    supports_dsa_cp_local_model_forward,
)
from vllm_ascend.models.deepseek_v4 import _prepare_dsa_cp_local_hidden
from vllm_ascend.ops.fused_moe.experts_selector import _select_dsa_cp_local_input_ids
from vllm_ascend.ops.rope_dsv4 import (
    _ROPE_STATE,
    RopeDataProxy,
    concatenate_rope_slices,
    concatenate_tensor_slices,
    resolve_rope_tensor,
)


def test_find_local_cache_plan_supports_nested_ubatch_metadata():
    plan = build_dsa_cp_local_cache_plan(258, 2, 0, [0, 129, 258])
    metadata = SimpleNamespace(
        req_metadata=SimpleNamespace(
            cp_metadata=SimpleNamespace(local_cache_plan=plan)
        )
    )
    nested_metadata = [{"layer": metadata}]

    assert find_dsa_cp_local_cache_plan(nested_metadata) is plan


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


def test_local_cache_cp_concatenates_layer_rope_tensor_ranges():
    cos = torch.arange(24).reshape(6, 4)
    sin = cos + 100
    local_valid_ranges = ((0, 2), (4, 6))

    local_cos = concatenate_tensor_slices(cos, local_valid_ranges)
    local_sin = concatenate_tensor_slices(sin, local_valid_ranges)

    assert torch.equal(local_cos, torch.cat((cos[0:2], cos[4:6])))
    assert torch.equal(local_sin, torch.cat((sin[0:2], sin[4:6])))


def test_resolve_rope_tensor_supports_tensor_and_proxy(monkeypatch):
    cos = torch.arange(24).reshape(6, 4)
    sin = cos + 100
    layer_name = "model.layers.0.self_attn"
    monkeypatch.setitem(_ROPE_STATE.layer_info, layer_name, ("config", ["c4"]))
    cos_proxy = RopeDataProxy({"config": {"c4": (cos, sin)}}, is_cos=True)
    sin_proxy = RopeDataProxy({"config": {"c4": (cos, sin)}}, is_cos=False)

    assert resolve_rope_tensor(cos, layer_name) is cos
    assert resolve_rope_tensor(sin, layer_name) is sin
    assert resolve_rope_tensor(cos_proxy, layer_name) is cos
    assert resolve_rope_tensor(sin_proxy, layer_name) is sin


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


def test_local_cache_cp_rejects_plan_with_empty_model_input_rank():
    plans = _plans(num_input_tokens=1, cp_size=64)

    assert all(not supports_dsa_cp_local_model_forward(plan) for plan in plans)


def test_local_cache_cp_builder_falls_back_for_short_request_with_empty_rank():
    builder = SimpleNamespace(
        enable_dsa_cp_local_cache=True,
        cp_size=64,
        cp_rank=0,
        num_actual_tokens=1,
    )

    plan = AscendDSACPMetadataBuilder._build_local_cache_plan(
        builder,
        num_input_tokens=1,
        query_start_loc=[0, 1],
    )

    assert plan is None


def test_local_cache_cp_accepts_plan_when_every_rank_has_model_input():
    plans = _plans(num_input_tokens=256, cp_size=2)

    assert all(supports_dsa_cp_local_model_forward(plan) for plan in plans)


def test_local_cache_plan_excludes_model_input_padding():
    builder = SimpleNamespace(
        enable_dsa_cp_local_cache=True,
        cp_size=4,
        cp_rank=3,
        num_actual_tokens=511,
    )

    plan = AscendDSACPMetadataBuilder._build_local_cache_plan(
        builder,
        num_input_tokens=512,
        query_start_loc=[0, 511],
    )

    assert plan is not None
    assert plan.local_valid_ranges == ((384, 511),)
    assert plan.local_num_tokens == 127
    assert plan.all_rank_num_tokens == (128, 128, 128, 127)


@pytest.mark.parametrize(
    ("rank", "expected_start", "expected_tokens"),
    [(0, 0, 1408), (1, 1408, 1408), (2, 2816, 1408), (3, 4224, 1403)],
)
def test_local_cache_model_input_selects_variable_rank_ranges(rank, expected_start, expected_tokens):
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=5627,
        cp_size=4,
        cp_rank=rank,
        query_start_loc=[0, 5627],
    )
    full_hidden = torch.arange(5628, dtype=torch.float32).view(-1, 1)

    local_hidden = _prepare_dsa_cp_local_hidden(full_hidden, plan)

    assert local_hidden.shape == (expected_tokens, 1)
    assert local_hidden.flatten().tolist() == list(
        range(expected_start, expected_start + expected_tokens)
    )


def test_compact_sample_tokens_supports_uneven_cp4_layout():
    full_hidden = torch.arange(5627, dtype=torch.float32).view(-1, 1)
    sample_indices = torch.tensor([0, 1407, 1408, 4223, 4224, 5626, 5626])
    contributions = []
    for rank in range(4):
        plan = build_dsa_cp_local_cache_plan(
            num_input_tokens=5627,
            cp_size=4,
            cp_rank=rank,
            query_start_loc=[0, 5627],
        )
        local_hidden = select_dsa_cp_local_tokens(full_hidden, plan)
        with (
            patch("vllm_ascend.attention.context_parallel.dsa_cp.dist.all_reduce"),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
                return_value=SimpleNamespace(device_group=None),
            ),
        ):
            contributions.append(
                compact_dsa_cp_sample_tokens(local_hidden, sample_indices, plan)
            )

    assert torch.equal(
        torch.stack(contributions).sum(dim=0), full_hidden[sample_indices.long()]
    )


def test_local_cache_model_input_preserves_multi_batch_ranges():
    query_start_loc = [0, 1, 2, 3]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=3,
        cp_size=2,
        cp_rank=0,
        query_start_loc=query_start_loc,
    )
    gathered_hidden = torch.arange(4, dtype=torch.float32).view(-1, 1)
    local_hidden = _prepare_dsa_cp_local_hidden(gathered_hidden, plan)

    expected = torch.cat(
        [gathered_hidden[start:end] for start, end in plan.local_valid_ranges]
    )
    assert len(plan.local_valid_ranges) == 2
    assert torch.equal(local_hidden, expected)


def test_local_cache_model_input_uses_current_chunk_boundaries():
    query_start_loc = [0, 129, 258]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=258,
        cp_size=2,
        cp_rank=1,
        query_start_loc=query_start_loc,
    )
    gathered_hidden = torch.arange(258, dtype=torch.float32).view(-1, 1)
    local_hidden = _prepare_dsa_cp_local_hidden(gathered_hidden, plan)

    expected = torch.cat(
        [gathered_hidden[start:end] for start, end in plan.local_valid_ranges]
    )
    assert torch.equal(local_hidden, expected)


def test_local_cache_layout_restores_multi_batch_global_token_order():
    query_start_loc = [0, 129, 258]
    full_tensor = torch.arange(258, dtype=torch.float32).view(-1, 1)
    plans = [
        build_dsa_cp_local_cache_plan(
            num_input_tokens=258,
            cp_size=2,
            cp_rank=rank,
            query_start_loc=query_start_loc,
        )
        for rank in range(2)
    ]
    rank_tensors = [select_dsa_cp_local_tokens(full_tensor, plan) for plan in plans]

    restored = restore_dsa_cp_global_tokens(rank_tensors, plans[0])

    assert any(len(ranges) > 1 for ranges in plans[0].rank_valid_ranges)
    assert torch.equal(restored, full_tensor)


def test_compact_sample_tokens_supports_multi_batch_chunk_ranges():
    query_start_loc = [0, 129, 258]
    full_hidden = torch.arange(258, dtype=torch.float32).view(-1, 1)
    sample_indices = torch.tensor([128, 257, 0], dtype=torch.int32)
    contributions = []
    for rank in range(2):
        plan = build_dsa_cp_local_cache_plan(258, 2, rank, query_start_loc)
        local_hidden = select_dsa_cp_local_tokens(full_hidden, plan)
        with (
            patch("vllm_ascend.attention.context_parallel.dsa_cp.dist.all_reduce"),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_tp_group",
                return_value=SimpleNamespace(device_group=None),
            ),
        ):
            contributions.append(
                compact_dsa_cp_sample_tokens(local_hidden, sample_indices, plan)
            )
    assert torch.equal(
        torch.stack(contributions).sum(dim=0), full_hidden[sample_indices.long()]
    )


def test_legacy_cp_model_input_is_unchanged():
    hidden_states = torch.arange(8)

    selected = _prepare_dsa_cp_local_hidden(hidden_states, None)

    assert selected is hidden_states


@pytest.mark.parametrize(
    ("rank", "expected_start", "expected_tokens"),
    [(0, 0, 1408), (1, 1408, 1408), (2, 2816, 1408), (3, 4224, 1403)],
)
def test_local_cache_hash_gating_input_ids_match_router_rows(
    rank, expected_start, expected_tokens
):
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=5627,
        cp_size=4,
        cp_rank=rank,
        query_start_loc=[0, 5627],
    )
    input_ids = torch.arange(5628)

    local_input_ids = _select_dsa_cp_local_input_ids(input_ids, plan)

    assert local_input_ids.shape[0] == expected_tokens
    assert local_input_ids.tolist() == list(
        range(expected_start, expected_start + expected_tokens)
    )


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
    assert window_plan.final_window_ranges == [(9088, 9216)]
    assert window_plan.input_slot_mapping.tolist() == list(range(1920, 2176))
    assert window_plan.final_window_slot_mapping.tolist() == list(range(9088, 9216))


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
    assert window_plan.final_window_ranges == [(64, 192), (256, 384)]


def test_swa_window_plan_uses_natural_rank_predecessor_and_request_tail():
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=5627,
        cp_size=4,
        cp_rank=3,
        query_start_loc=[0, 5627],
    )
    slot_mapping = torch.arange(10_000, 10_000 + 5627)

    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 5627],
        num_actual_tokens=5627,
        slot_mapping=slot_mapping,
    )

    assert window_plan.valid_ranges == [(4224, 5627)]
    assert window_plan.halo_ranges == [(4096, 4224)]
    assert window_plan.input_ranges == [(4096, 5627)]
    assert window_plan.final_window_ranges == [(5499, 5627)]
    assert window_plan.input_slot_mapping.tolist() == list(
        range(10_000 + 4096, 10_000 + 5627)
    )
    assert window_plan.final_window_slot_mapping.tolist() == list(
        range(10_000 + 5499, 10_000 + 5627)
    )


def test_draft_swa_plans_use_spec_slot_mapping():
    local_cache_plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=256,
        cp_size=2,
        cp_rank=1,
        query_start_loc=[0, 256],
    )
    builder = SimpleNamespace(slot_mapping=torch.arange(256))
    spec_slot_mapping = torch.arange(10_000, 10_256)

    local_slots, valid_start, valid_end = (
        AscendDSACPMetadataBuilder._build_swa_local_slot_mapping(
            builder,
            local_cache_plan,
            256,
            spec_slot_mapping,
        )
    )
    window_plan = AscendDSACPMetadataBuilder._build_swa_window_plan(
        builder,
        local_cache_plan,
        [0, 256],
        256,
        spec_slot_mapping,
    )

    assert local_slots.tolist() == list(range(10_128, 10_256))
    assert (valid_start, valid_end) == (0, 128)
    assert window_plan is not None
    assert window_plan.input_slot_mapping.tolist() == list(range(10_000, 10_256))
    assert window_plan.final_window_slot_mapping.tolist() == list(
        range(10_128, 10_256)
    )


def test_swa_final_window_plan_is_per_request_for_multi_batch():
    query_start_loc = [0, 64, 364]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=364,
        cp_size=2,
        cp_rank=1,
        query_start_loc=query_start_loc,
    )

    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=364,
    )
    final_hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=window_plan.final_window_ranges,
        local_cache_plan=plan,
        num_actual_tokens=364,
    )

    assert window_plan.final_window_ranges == [(0, 64), (236, 364)]
    assert window_plan.final_window_slot_mapping.tolist() == [
        *range(0, 64),
        *range(236, 364),
    ]
    assert final_hidden_plan.num_input_tokens == 192


def test_swa_final_window_plan_updates_only_current_chunk_tokens():
    # This chunk contains absolute request positions [2048, 2080).  The
    # previous chunk's 96 surviving window entries are already in cache, so
    # only these 32 current slots need to be rebuilt on every rank.
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=32,
        cp_size=1,
        cp_rank=0,
        query_start_loc=[0, 32],
    )
    slot_mapping = torch.arange(500, 532)

    window_plan = build_dsa_cp_swa_window_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 32],
        num_actual_tokens=32,
        slot_mapping=slot_mapping,
    )

    assert window_plan.final_window_ranges == [(0, 32)]
    assert torch.equal(window_plan.final_window_slot_mapping, slot_mapping)


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
    assert compressor_plan.slot_mapping.tolist() == [-1, 1032, 1033, -1]
    assert compressor_plan.valid_output_mask.tolist() == [False, True, True, False]
    assert compressor_plan.valid_output_indices.tolist() == [1, 2]
    assert compressor_plan.output_indices.tolist() == [31, 32, 33, -1]
    assert compressor_plan.compressed_positions.tolist() == [124, 128, 132, 132]
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
    input_positions = torch.cat([torch.arange(132), torch.arange(1025, 1157)])
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
    assert compressor_plan.compressed_positions[:2].tolist() == [1024, 1028]
    assert compressor_plan.input_indices.tolist() == list(range(132, 260))
    assert compressor_plan.input_query_start_loc.tolist() == [0, 128]
    assert compressor_plan.request_indices.tolist() == [1]
    assert compressor_plan.start_pos_offsets.tolist() == [0]


def test_c128_local_compressor_slot_plan_starts_at_current_group():
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
        overlap_tokens=0,
    )

    assert compressor_plan.input_ranges == [(2048, 2176)]
    assert compressor_plan.valid_ranges == [(2048, 2176)]
    assert compressor_plan.overlap_ranges == [(2048, 2048)]
    assert compressor_plan.slot_mapping.tolist() == [3016, -1]
    assert compressor_plan.valid_output_mask.tolist() == [True, False]
    assert compressor_plan.valid_output_indices.tolist() == [0]
    assert compressor_plan.output_indices.tolist() == [16, -1]
    assert compressor_plan.compressed_positions.tolist() == [2048, 2048]
    assert compressor_plan.input_indices.tolist() == list(range(2048, 2176))
    assert compressor_plan.input_query_start_loc.tolist() == [0, 128]
    assert compressor_plan.request_indices.tolist() == [0]
    assert compressor_plan.start_pos_offsets.tolist() == [2048]


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
        overlap_tokens=0,
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

    assert compressor_plan.slot_mapping.shape == (4, 2)
    assert compressor_plan.slot_mapping.tolist() == [
        [-1, -1],
        [32, 132],
        [33, 133],
        [-1, -1],
    ]
    assert compressor_plan.output_indices.tolist() == [31, 32, 33, -1]
    assert compressor_plan.compressed_positions.tolist() == [124, 128, 132, 132]
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


def test_state_broadcast_plan_adds_all_crossed_boundaries_before_tail():
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=24,
        cp_size=2,
        cp_rank=0,
        unit_size=4,
    )
    state_block_table = torch.arange(1000, 1018, dtype=torch.int32).reshape(1, 18)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 24],
        num_actual_tokens=24,
        input_positions=torch.arange(12, 36),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=2,
        cache_transfer_granularity=16,
    )

    assert state_plan.state_block_indices.tolist() == [6, 7, 14, 15, 16, 17]
    assert state_plan.state_block_source_ranks.tolist() == [0, 0, 1, 1, 1, 1]
    assert state_plan.state_request_indices.tolist() == [0] * 6
    assert [
        (run.source_rank, run.first_block_id, run.last_block_id)
        for run in state_plan.broadcast_runs
    ] == [
        (0, 1006, 1008),
        (1, 1014, 1018),
    ]


def test_state_broadcast_plan_deduplicates_boundary_and_tail_blocks():
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=16,
        cp_size=2,
        cp_rank=0,
        unit_size=4,
    )
    state_block_table = torch.arange(2000, 2008, dtype=torch.int32).reshape(1, 8)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 16],
        num_actual_tokens=16,
        input_positions=torch.arange(16),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=2,
        cache_transfer_granularity=16,
    )

    assert state_plan.state_block_indices.tolist() == [6, 7]
    assert state_plan.state_block_source_ranks.tolist() == [1, 1]


def test_state_broadcast_plan_handles_batched_chunk_boundaries():
    query_start_loc = [0, 8, 16]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=16,
        cp_size=2,
        cp_rank=0,
        unit_size=4,
        query_start_loc=query_start_loc,
    )
    state_block_table = torch.arange(3000, 3036, dtype=torch.int32).reshape(2, 18)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=16,
        input_positions=torch.cat((torch.arange(12, 20), torch.arange(28, 36))),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=2,
        cache_transfer_granularity=16,
    )

    assert state_plan.state_block_indices.tolist() == [6, 7, 8, 9, 14, 15, 16, 17]
    assert state_plan.state_block_source_ranks.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert state_plan.state_request_indices.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_state_boundary_granularity_matches_store_page_families():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        cache_config=SimpleNamespace(block_size=128),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[4, 128, 0])
        ),
    )

    assert get_dsa_cp_state_boundary_granularity(config) == 16384
    config.kv_transfer_config.is_kv_producer = False
    assert get_dsa_cp_state_boundary_granularity(config) == 0


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
    query_start_loc = [0, 130]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=130, cp_size=2, cp_rank=1, query_start_loc=query_start_loc
    )
    state_block_table = torch.arange(1000, 1000 + 65, dtype=torch.int32).reshape(1, 65)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=130,
        input_positions=torch.arange(1920, 2050),
        state_block_table=state_block_table,
        compress_ratio=128,
        state_block_size=32,
    )

    assert state_plan.source_ranks.tolist() == [1]
    assert state_plan.state_block_source_ranks.tolist() == [1, 1, 1, 1, 1]
    assert state_plan.state_block_indices.tolist() == [60, 61, 62, 63, 64]
    assert state_plan.state_block_ids.tolist() == [1060, 1061, 1062, 1063, 1064]
    assert state_plan.state_valid_mask.tolist() == [True] * 5
    assert state_plan.state_request_indices.tolist() == [0] * 5


def test_state_broadcast_plan_selects_c4_tail_state_block():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=10, cp_size=2, cp_rank=1)
    state_block_table = torch.arange(2000, 2000 + 257, dtype=torch.int32).reshape(1, 257)

    state_plan = build_dsa_cp_state_broadcast_plan(
        local_cache_plan=plan,
        query_start_loc=[0, 10],
        num_actual_tokens=10,
        input_positions=torch.arange(2040, 2050),
        state_block_table=state_block_table,
        compress_ratio=4,
        state_block_size=8,
    )

    assert state_plan.source_ranks.tolist() == [1]
    assert state_plan.local_request_indices.tolist() == [0]
    assert state_plan.tail_token_offsets.tolist() == [9]
    assert state_plan.state_block_source_ranks.tolist() == [1, 1]
    assert state_plan.state_block_indices.tolist() == [255, 256]
    assert state_plan.state_block_ids.tolist() == [2255, 2256]
    assert state_plan.state_valid_mask.tolist() == [True, True]


def test_state_broadcast_plan_uses_request_local_positions_for_state_blocks():
    query_start_loc = [0, 132, 264]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=264, cp_size=3, cp_rank=1, query_start_loc=query_start_loc
    )
    state_block_table = torch.arange(3000, 3000 + 2 * 17, dtype=torch.int32).reshape(2, 17)

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
    assert state_plan.state_block_indices.tolist() == [16, 16]
    assert state_plan.state_block_ids.tolist() == [3016, 3033]
    assert state_plan.state_valid_mask.tolist() == [True, True]
    assert state_plan.state_request_indices.tolist() == [0, 1]


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


def test_state_broadcast_helper_coalesces_contiguous_state_blocks(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    state_cache = torch.zeros((8, 2, 1, 3), dtype=torch.float32)
    plan = DSACPStateBroadcastPlan(
        source_ranks=torch.tensor([0, 0], dtype=torch.int32),
        local_request_indices=torch.tensor([], dtype=torch.long),
        tail_token_offsets=torch.tensor([127, 255], dtype=torch.long),
        state_block_ids=torch.tensor([1, 2, 3, 5, 6], dtype=torch.int32),
        state_block_indices=torch.tensor([4, 5, 6, 0, 1], dtype=torch.long),
        state_valid_mask=torch.ones(5, dtype=torch.bool),
        state_block_source_ranks=torch.zeros(5, dtype=torch.int32),
        state_request_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    calls = []

    def fake_broadcast(buffer, src, group):
        calls.append((src, group, tuple(buffer.shape), buffer.data_ptr()))
        buffer.fill_(float(len(calls)))

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fake_broadcast,
    )

    impl._broadcast_dsa_cp_state_blocks(state_cache, plan)

    assert [call[:3] for call in calls] == [
        (10, impl.tp_group.device_group, (3, 2, 1, 3)),
        (10, impl.tp_group.device_group, (2, 2, 1, 3)),
    ]
    assert calls[0][3] == state_cache[1:4].data_ptr()
    assert calls[1][3] == state_cache[5:7].data_ptr()
    assert torch.all(state_cache[1:4] == 1.0)
    assert torch.all(state_cache[5:7] == 2.0)
    assert torch.all(state_cache[[0, 4, 7]] == 0.0)


def test_state_broadcast_helper_packs_noncontiguous_state_cache(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    state_cache_storage = torch.zeros((16, 2, 1, 3), dtype=torch.float32)
    state_cache = state_cache_storage[::2]
    assert not state_cache.is_contiguous()
    plan = DSACPStateBroadcastPlan(
        source_ranks=torch.tensor([0], dtype=torch.int32),
        local_request_indices=torch.tensor([], dtype=torch.long),
        tail_token_offsets=torch.tensor([127], dtype=torch.long),
        state_block_ids=torch.tensor([1, 2], dtype=torch.int32),
        state_block_indices=torch.tensor([0, 1], dtype=torch.long),
        state_valid_mask=torch.ones(2, dtype=torch.bool),
        state_block_source_ranks=torch.zeros(2, dtype=torch.int32),
        state_request_indices=torch.zeros(2, dtype=torch.long),
    )

    def fake_broadcast(buffer, src, group):
        assert buffer.is_contiguous()
        buffer.fill_(5.0)

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fake_broadcast,
    )

    impl._broadcast_dsa_cp_state_blocks(state_cache, plan)

    assert torch.all(state_cache[1:3] == 5.0)
    assert torch.all(state_cache[[0, 3, 4, 5, 6, 7]] == 0.0)


def test_state_broadcast_helper_splits_noncontiguous_physical_blocks(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 2
    impl.tp_rank = 1
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11])

    state_cache = torch.zeros((6, 2, 1, 3), dtype=torch.float32)
    plan = DSACPStateBroadcastPlan(
        source_ranks=torch.tensor([0], dtype=torch.int32),
        local_request_indices=torch.tensor([], dtype=torch.long),
        tail_token_offsets=torch.tensor([127], dtype=torch.long),
        state_block_ids=torch.tensor([1, 3, 4], dtype=torch.int32),
        state_block_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        state_valid_mask=torch.ones(3, dtype=torch.bool),
        state_block_source_ranks=torch.zeros(3, dtype=torch.int32),
        state_request_indices=torch.zeros(3, dtype=torch.long),
    )
    calls = []

    def fake_broadcast(buffer, src, group):
        calls.append((src, group, tuple(buffer.shape)))
        buffer.fill_(9.0)

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fake_broadcast,
    )

    impl._broadcast_dsa_cp_state_blocks(state_cache, plan)

    assert calls == [
        (10, impl.tp_group.device_group, (1, 2, 1, 3)),
        (10, impl.tp_group.device_group, (2, 2, 1, 3)),
    ]
    assert torch.all(state_cache[[1, 3, 4]] == 9.0)
    assert torch.all(state_cache[[0, 2, 5]] == 0.0)


def test_state_broadcast_helper_keeps_precomputed_run_order(monkeypatch):
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.tp_size = 3
    impl.tp_rank = 2
    impl.tp_group = SimpleNamespace(device_group=object(), ranks=[10, 11, 12])

    state_cache = torch.zeros((8, 2, 1, 3), dtype=torch.float32)
    plan = DSACPStateBroadcastPlan(
        source_ranks=torch.tensor([1, 0], dtype=torch.int32),
        local_request_indices=torch.tensor([], dtype=torch.long),
        tail_token_offsets=torch.tensor([127, 255], dtype=torch.long),
        state_block_ids=torch.tensor([5, 1, 3, 6], dtype=torch.int32),
        state_block_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        state_valid_mask=torch.ones(4, dtype=torch.bool),
        state_block_source_ranks=torch.tensor([1, 0, 0, 1], dtype=torch.int32),
        state_request_indices=torch.tensor([0, 1, 1, 0], dtype=torch.long),
    )
    calls = []

    def fake_broadcast(buffer, src, group):
        calls.append((src, group, tuple(buffer.shape)))
        buffer.fill_(float(src))

    monkeypatch.setattr(
        "vllm_ascend.attention.context_parallel.dsa_cp.dist.broadcast",
        fake_broadcast,
    )

    impl._broadcast_dsa_cp_state_blocks(state_cache, plan)

    assert calls == [
        (11, impl.tp_group.device_group, (1, 2, 1, 3)),
        (10, impl.tp_group.device_group, (1, 2, 1, 3)),
        (10, impl.tp_group.device_group, (1, 2, 1, 3)),
        (11, impl.tp_group.device_group, (1, 2, 1, 3)),
    ]
    assert torch.all(state_cache[[1, 3]] == 10.0)
    assert torch.all(state_cache[[5, 6]] == 11.0)
    assert torch.all(state_cache[[0, 2, 4, 7]] == 0.0)


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


def test_hidden_input_assembly_combines_adjacent_rank_halos():
    plan = build_dsa_cp_local_cache_plan(num_input_tokens=384, cp_size=2, cp_rank=1)
    hidden_plan = build_dsa_cp_hidden_input_plan(
        input_ranges=[(64, 256)],
        local_cache_plan=plan,
        num_actual_tokens=384,
    )
    hidden_states_local = torch.arange(192, 384).view(192, 1).to(torch.float32)
    first_halo = torch.arange(64, 128).view(64, 1).to(torch.float32)
    second_halo = torch.arange(128, 192).view(64, 1).to(torch.float32)

    assembled = AscendDSACPImpl._assemble_dsa_cp_hidden_input(
        AscendDSACPImpl,
        hidden_plan,
        hidden_states_local,
        ([first_halo, second_halo], [(64, 128), (128, 192)]),
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


def test_c4_chunk_prefill_compressor_plan_uses_persisted_state():
    # The previous chunk's compressor state is already persisted. The current
    # chunk starts from its first token without restoring raw hidden states for
    # absolute positions [2048, 2050).
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
    assert compressor_plan.prefix_lengths.tolist() == [0]
    assert compressor_plan.current_start_positions.tolist() == [2050]
    assert compressor_plan.has_prefix_hidden is False
    assert compressor_plan.start_pos_offsets.tolist() == [0]
    assert compressor_plan.input_query_start_loc.tolist() == [0, 8]
    assert compressor_plan.slot_mapping.tolist() == [4000, 4001, -1]
    assert compressor_plan.valid_output_mask.tolist() == [True, True, False]
    assert compressor_plan.output_indices.tolist() == [0, 1, -1]
    assert compressor_plan.compressed_positions.tolist() == [2048, 2052, 2052]


def test_cp8_chunk_prefill_uses_compressor_metadata_slots():
    num_input_tokens = 1245
    input_positions = torch.arange(10000, 10000 + num_input_tokens)
    query_token_slots = torch.arange(10000, 10000 + num_input_tokens // 4 + 1)
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=num_input_tokens,
        cp_size=8,
        cp_rank=0,
    )
    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=query_token_slots,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, num_input_tokens],
        num_actual_tokens=num_input_tokens,
        overlap_tokens=4,
    )

    # Deliberately differ from the ordinary token slots above. Later chunks
    # must address compressed cache positions produced by compressor metadata.
    metadata_slots = torch.arange(2500, 2500 + num_input_tokens // 4 + 1)
    resolved_plan = materialize_dsa_cp_compressor_slot_plan(
        compressor_plan,
        metadata_slots,
    )

    assert plan.all_rank_num_tokens == (256, 256, 128, 128, 128, 128, 128, 93)
    assert resolved_plan.slot_mapping[:64].tolist() == list(range(2500, 2564))
    assert resolved_plan.slot_mapping[-1].item() == -1
    assert resolved_plan.all_rank_slot_mappings[0].tolist() == list(
        range(2500, 2564)
    )
    assert resolved_plan.all_rank_slot_mappings[7].tolist() == list(
        range(2788, 2811)
    )


def test_precompute_compressor_metadata_only_for_new_cp_prefill_producer_eager():
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True,
            is_kv_consumer=False,
        ),
    )

    assert should_precompute_dsa_cp_compressor_metadata(
        config,
        has_prefill=True,
        local_cache_plan=object(),
        compressor_slot_plan=object(),
    )


@pytest.mark.parametrize(
    (
        "has_prefill",
        "has_local_cache_plan",
        "has_compressor_slot_plan",
        "enforce_eager",
        "is_kv_producer",
        "is_kv_consumer",
    ),
    [
        (False, True, True, True, True, False),
        (True, False, True, True, True, False),
        (True, True, False, True, True, False),
        (True, True, True, False, True, False),
        (True, True, True, True, False, True),
        (True, True, True, True, True, True),
        (True, True, True, True, False, False),
    ],
)
def test_precompute_compressor_metadata_preserves_other_paths(
    has_prefill,
    has_local_cache_plan,
    has_compressor_slot_plan,
    enforce_eager,
    is_kv_producer,
    is_kv_consumer,
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=is_kv_producer,
            is_kv_consumer=is_kv_consumer,
        ),
    )

    assert not should_precompute_dsa_cp_compressor_metadata(
        config,
        has_prefill=has_prefill,
        local_cache_plan=object() if has_local_cache_plan else None,
        compressor_slot_plan=object() if has_compressor_slot_plan else None,
    )


def test_precompute_compressor_metadata_requires_pd_configuration():
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        kv_transfer_config=None,
    )

    assert not should_precompute_dsa_cp_compressor_metadata(
        config,
        has_prefill=True,
        local_cache_plan=object(),
        compressor_slot_plan=object(),
    )


def test_c128_chunk_prefill_compressor_plan_uses_current_chunk():
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
        overlap_tokens=0,
    )

    assert compressor_plan.input_ranges == [(0, 128)]
    assert compressor_plan.prefix_lengths.tolist() == [0]
    assert compressor_plan.has_prefix_hidden is False
    assert compressor_plan.start_pos_offsets.tolist() == [0]
    assert compressor_plan.input_query_start_loc.tolist() == [0, 128]
    assert compressor_plan.slot_mapping.tolist() == [5000, -1]
    assert compressor_plan.valid_output_mask.tolist() == [True, False]
    assert compressor_plan.output_indices.tolist() == [0, -1]
    assert compressor_plan.compressed_positions.tolist() == [2048, 2048]


@pytest.mark.parametrize(
    (
        "compress_ratio",
        "overlap_tokens",
        "expected_input_start",
        "expected_start_pos",
    ),
    [(4, 4, 1402, 3452), (128, 0, 1406, 3456)],
)
def test_chunk_prefill_later_rank_compressor_input_is_group_aligned(
    compress_ratio, overlap_tokens, expected_input_start, expected_start_pos
):
    # The chunk begins at an unaligned absolute position. Rank 1 therefore
    # cannot derive its overlap by subtracting a fixed number of flattened
    # tokens: that would start at 3454 for C4. That value points the
    # compressor at the wrong persisted state.
    num_input_tokens = 2816
    input_positions = torch.arange(2050, 2050 + num_input_tokens)
    slot_mapping = torch.arange(
        min(num_input_tokens, num_input_tokens // compress_ratio + 1)
    )
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=num_input_tokens, cp_size=2, cp_rank=1
    )

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=compress_ratio,
        local_cache_plan=plan,
        query_start_loc=[0, num_input_tokens],
        num_actual_tokens=num_input_tokens,
        overlap_tokens=overlap_tokens,
    )

    assert compressor_plan.valid_ranges == [(1408, 2816)]
    assert compressor_plan.input_ranges == [(expected_input_start, 2816)]
    assert compressor_plan.overlap_ranges == [(expected_input_start, 1408)]
    assert compressor_plan.start_pos_offsets.tolist() == [
        expected_start_pos - 2050
    ]
    assert expected_start_pos % compress_ratio == 0


def test_chunk_prefill_multi_batch_compressor_padding_is_appended_at_batch_tail():
    input_positions = torch.cat(
        [torch.arange(2050, 2058), torch.arange(1025, 1033)]
    )
    slot_mapping = torch.arange(6000, 6004)
    query_start_loc = [0, 8, 16]
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=16,
        cp_size=1,
        cp_rank=0,
        query_start_loc=query_start_loc,
    )

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=query_start_loc,
        num_actual_tokens=16,
        overlap_tokens=4,
    )

    # Both requests inherit their persisted compressor state and consume only
    # their eight current-chunk tokens. The ABI requires
    # min(16, 16 // 4 + 2) == 6 rows. Four valid rows stay contiguous and the
    # two unused capacity rows are appended after the whole local batch.
    assert compressor_plan.prefix_lengths.tolist() == [0, 0]
    assert compressor_plan.input_query_start_loc.dtype == torch.int32
    assert compressor_plan.input_query_start_loc.tolist() == [0, 8, 16]
    assert compressor_plan.request_indices.tolist() == [0, 1]
    assert compressor_plan.start_pos_offsets.tolist() == [0, 0]
    assert compressor_plan.slot_mapping.tolist() == [
        6000,
        6001,
        6002,
        6003,
        -1,
        -1,
    ]
    assert compressor_plan.valid_output_mask.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert compressor_plan.output_indices.tolist() == [0, 1, 2, 3, -1, -1]
    assert compressor_plan.compressed_positions.tolist() == [
        2048,
        2052,
        1024,
        1028,
        1028,
        1028,
    ]


def test_c4_aligned_local_compressor_plan_reserves_abi_boundary_row():
    num_input_tokens = 1408
    plan = build_dsa_cp_local_cache_plan(
        num_input_tokens=num_input_tokens,
        cp_size=1,
        cp_rank=0,
    )

    compressor_plan = build_dsa_cp_local_compressor_slot_plan(
        input_positions=torch.arange(num_input_tokens),
        slot_mapping=torch.arange(num_input_tokens // 4),
        compress_ratio=4,
        local_cache_plan=plan,
        query_start_loc=[0, num_input_tokens],
        num_actual_tokens=num_input_tokens,
        overlap_tokens=4,
    )

    assert compressor_plan.input_query_start_loc.tolist() == [0, num_input_tokens]
    assert compressor_plan.compressed_positions.shape == (353,)
    assert compressor_plan.slot_mapping.shape == (353,)
    assert compressor_plan.valid_output_mask[:352].all()
    assert not compressor_plan.valid_output_mask[-1]
    assert compressor_plan.output_indices[-1] == -1
    assert compressor_plan.slot_mapping[-1] == -1
    assert compressor_plan.compressed_positions[-2:].tolist() == [1404, 1404]


def test_compressor_hidden_input_assembly_uses_only_current_chunk():
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
    hidden_states_local = torch.arange(10, 18).view(8, 1).to(torch.float32)

    assembled = AscendDSACPImpl._assemble_dsa_cp_hidden_input(
        AscendDSACPImpl,
        hidden_input_plan,
        hidden_states_local,
        None,
    )

    assert assembled.flatten().tolist() == list(range(10, 18))
