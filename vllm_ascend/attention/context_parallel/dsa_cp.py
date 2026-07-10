import math
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs as ascend_envs
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import (
    concatenate_rope_slices,
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa,
)
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_dsa_cp_with_o_proj_tp,
    get_ascend_device_type,
    olora_tp_enable,
)


DSACP_LOCAL_CACHE_UNIT_SIZE = 128

def hadamard_transform_ref(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    scale: float = 1.0,  # type: ignore[assignment]
):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)


def _has_prefill(attn_state: AscendAttentionState) -> bool:
    return attn_state not in {
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    }


@dataclass
class DSACPLocalCachePlan:
    """Local-cache CP plan for the opt-in DeepSeek DSA prefill path."""

    enabled: bool
    cp_size: int
    cp_rank: int
    query_start_loc: tuple[int, ...]
    rank_request_ranges: tuple[tuple[tuple[int, int], ...], ...] # ranks(requests(start,end)) without padding
    rank_valid_ranges: tuple[tuple[tuple[int, int], ...], ...]
    local_request_ranges: tuple[tuple[int, int], ...]
    local_valid_ranges: tuple[tuple[int, int], ...]
    local_offsets: tuple[int, ...]
    all_rank_num_tokens: tuple[int, ...]
    local_start: int
    local_end: int
    tokens_per_rank: int
    num_tokens_pad: int
    unit_size: int
    local_num_tokens: int


@dataclass
class DSACPSWAWindowPlan:
    """Per-request SWA owner ranges and 128-token halo ranges."""

    input_ranges: list[tuple[int, int]]
    valid_ranges: list[tuple[int, int]]
    halo_ranges: list[tuple[int, int]]
    all_rank_valid_token_counts: tuple[int, ...]
    all_rank_slot_mappings: tuple[torch.Tensor, ...]


@dataclass
class DSACPCompressorSlotPlan:
    """Local compressor input ranges and owner-only cache slot mapping."""

    input_ranges: list[tuple[int, int]]
    valid_ranges: list[tuple[int, int]]
    overlap_ranges: list[tuple[int, int]]
    slot_mapping: torch.Tensor
    valid_output_mask: torch.Tensor
    output_indices: torch.Tensor
    compressed_positions: torch.Tensor
    input_indices: torch.Tensor
    input_query_start_loc: torch.Tensor
    request_indices: torch.Tensor
    request_indices_cpu: torch.Tensor
    request_ids: list[str] | None
    start_pos_offsets: torch.Tensor
    prefix_lengths: torch.Tensor
    current_start_positions: torch.Tensor
    has_prefix_hidden: bool
    all_rank_valid_output_counts: tuple[int, ...]
    all_rank_slot_mappings: tuple[torch.Tensor, ...]


@dataclass
class DSACPStateBroadcastPlan:
    """Per-request final-state owner and state-block selection plan."""

    source_ranks: torch.Tensor
    local_request_indices: torch.Tensor
    tail_token_offsets: torch.Tensor
    state_block_ids: torch.Tensor
    state_block_indices: torch.Tensor
    state_valid_mask: torch.Tensor


@dataclass
class DSACPHiddenInputPlan:
    """Source ranges for assembling local-cache CP hidden inputs."""

    input_ranges: list[tuple[int, int]]
    local_source_ranges: list[tuple[int, int]]
    local_read_ranges: list[tuple[int, int]]
    local_output_ranges: list[tuple[int, int]]
    halo_source_ranges: list[tuple[int, int]]
    halo_output_ranges: list[tuple[int, int]]
    num_input_tokens: int


def _get_dsa_cp_local_range(
    num_input_tokens: int,
    cp_size: int,
    cp_rank: int,
    unit_size: int,
) -> tuple[int, int]:
    num_units = math.ceil(num_input_tokens / unit_size)
    base_units = num_units // cp_size
    remainder_units = num_units % cp_size
    local_units = base_units + int(cp_rank < remainder_units)
    local_start_units = cp_rank * base_units + min(cp_rank, remainder_units)
    local_start = local_start_units * unit_size
    return local_start, local_start + local_units * unit_size


def _ceil_to_unit(num_tokens: int, unit_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return math.ceil(num_tokens / unit_size) * unit_size


def _logical_padded_offset_to_real(
    logical_offset: int,
    request_padded_start: int,
    request_start: int,
    request_len: int,
) -> int:
    request_offset = max(0, logical_offset - request_padded_start)
    return request_start + min(request_offset, request_len)



def _filter_non_empty_ranges(
    ranges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple((start, end) for start, end in ranges if start < end)


def _build_local_offsets(ranges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for start, end in ranges:
        offsets.append(cursor)
        cursor += end - start
    return tuple(offsets)


def _sum_ranges(ranges: tuple[tuple[int, int], ...]) -> int:
    return sum(max(0, end - start) for start, end in ranges)


def _select_indexer_hidden_states_full(
    hidden_states_full: torch.Tensor | None,
    hidden_states_cache: torch.Tensor,
    use_local_cache_prefill: bool,
) -> torch.Tensor | None:
    """Keep legacy and local-cache CP indexer inputs isolated."""
    if use_local_cache_prefill:
        return hidden_states_full
    return hidden_states_cache


def _find_local_range_offset(
    local_cache_plan: DSACPLocalCachePlan,
    token_offset: int,
) -> tuple[int, int, int] | None:
    for (range_start, range_end), local_offset in zip(
        local_cache_plan.local_valid_ranges, local_cache_plan.local_offsets
    ):
        if range_start <= token_offset < range_end:
            return range_start, range_end, local_offset
    return None


def _next_local_range_start(
    local_cache_plan: DSACPLocalCachePlan,
    token_offset: int,
    range_end: int,
) -> int:
    next_start = range_end
    for local_start, local_end in local_cache_plan.local_valid_ranges:
        if token_offset < local_start:
            next_start = min(next_start, local_start)
        elif local_start <= token_offset < local_end:
            next_start = token_offset
            break
    return next_start


def get_dsa_cp_all_rank_token_counts(
    local_cache_plan: DSACPLocalCachePlan,
    num_actual_tokens: int | None = None,
) -> tuple[int, ...]:
    if num_actual_tokens is None:
        return local_cache_plan.all_rank_num_tokens
    token_counts = []
    for rank_ranges in local_cache_plan.rank_valid_ranges:
        token_counts.append(
            sum(max(0, min(end, num_actual_tokens) - min(start, num_actual_tokens)) for start, end in rank_ranges)
        )
    return tuple(token_counts)


def build_dsa_cp_hidden_input_plan(
    input_ranges: list[tuple[int, int]],
    local_cache_plan: DSACPLocalCachePlan,
    num_actual_tokens: int,
) -> DSACPHiddenInputPlan:
    """Plan hidden-state assembly from current local hidden and halo cache.

    Ranges use flattened-batch token offsets. ``local_source_ranges`` are read
    from this rank's compact local hidden tensor. ``halo_source_ranges`` must be
    provided by cross-rank gather or the per-layer hidden-state halo cache.
    Output ranges address the concatenated input tensor assembled from
    ``input_ranges`` in order.
    """

    clipped_input_ranges: list[tuple[int, int]] = []
    local_source_ranges: list[tuple[int, int]] = []
    local_read_ranges: list[tuple[int, int]] = []
    local_output_ranges: list[tuple[int, int]] = []
    halo_source_ranges: list[tuple[int, int]] = []
    halo_output_ranges: list[tuple[int, int]] = []
    output_offset = 0

    for range_start, range_end in input_ranges:
        range_start = max(0, min(range_start, num_actual_tokens))
        range_end = max(0, min(range_end, num_actual_tokens))
        if range_start >= range_end:
            continue

        clipped_input_ranges.append((range_start, range_end))
        cursor = range_start
        while cursor < range_end:
            local_segment = _find_local_range_offset(local_cache_plan, cursor)
            if local_segment is None:
                segment_end = _next_local_range_start(local_cache_plan, cursor, range_end)
                if segment_end <= cursor:
                    segment_end = range_end
                halo_source_ranges.append((cursor, segment_end))
                halo_output_ranges.append((output_offset, output_offset + segment_end - cursor))
            else:
                local_start, local_end, local_offset = local_segment
                segment_end = min(range_end, local_end)
                read_start = local_offset + cursor - local_start
                read_end = read_start + segment_end - cursor
                local_source_ranges.append((cursor, segment_end))
                local_read_ranges.append((read_start, read_end))
                local_output_ranges.append((output_offset, output_offset + segment_end - cursor))
            output_offset += segment_end - cursor
            cursor = segment_end

    return DSACPHiddenInputPlan(
        input_ranges=clipped_input_ranges,
        local_source_ranges=local_source_ranges,
        local_read_ranges=local_read_ranges,
        local_output_ranges=local_output_ranges,
        halo_source_ranges=halo_source_ranges,
        halo_output_ranges=halo_output_ranges,
        num_input_tokens=output_offset,
    )


def build_dsa_cp_state_broadcast_plan(
    local_cache_plan: DSACPLocalCachePlan,
    query_start_loc: list[int],
    num_actual_tokens: int,
    input_positions: torch.Tensor | None = None,
    state_block_table: torch.Tensor | None = None,
    compress_ratio: int = 1,
    state_block_size: int = 1,
) -> DSACPStateBroadcastPlan:
    """Build final-state broadcast ownership for current flattened chunk.

    When state block inputs are provided, the plan also identifies the state
    cache block that stores each request's final compressor/indexer state.
    """

    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if state_block_size <= 0:
        raise ValueError(f"state_block_size must be positive, got {state_block_size}")

    source_ranks: list[int] = []
    local_request_indices: list[int] = []
    tail_token_offsets: list[int] = []
    state_block_indices: list[int] = []
    state_valid_mask: list[bool] = []

    has_state_blocks = input_positions is not None and state_block_table is not None
    max_state_blocks = state_block_table.shape[1] if state_block_table is not None else 0

    for req_idx, (req_start, req_end) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
        req_start = min(req_start, num_actual_tokens)
        req_end = min(req_end, num_actual_tokens)
        if req_start >= req_end:
            source_ranks.append(-1)
            tail_token_offsets.append(-1)
            state_block_indices.append(-1)
            state_valid_mask.append(False)
            continue

        tail_token_offset = req_end - 1
        tail_owner_rank = -1
        for rank, rank_ranges in enumerate(local_cache_plan.rank_valid_ranges):
            if req_idx >= len(rank_ranges):
                continue
            rank_start, rank_end = rank_ranges[req_idx]
            if rank_start <= tail_token_offset < rank_end:
                tail_owner_rank = rank
                break

        source_ranks.append(tail_owner_rank)
        tail_token_offsets.append(tail_token_offset)
        if tail_owner_rank == local_cache_plan.cp_rank:
            local_request_indices.append(req_idx)

        state_block_index = -1
        state_valid = False
        if has_state_blocks:
            tail_position = int(input_positions[tail_token_offset].item())
            state_index = tail_position // compress_ratio
            state_block_index = state_index // state_block_size
            state_valid = tail_owner_rank >= 0 and state_block_index < max_state_blocks
        state_block_indices.append(state_block_index if state_valid else -1)
        state_valid_mask.append(state_valid)

    source_rank_tensor = torch.tensor(source_ranks, dtype=torch.int32)
    local_request_tensor = torch.tensor(local_request_indices, dtype=torch.long)
    tail_offset_tensor = torch.tensor(tail_token_offsets, dtype=torch.long)
    state_block_index_tensor = torch.tensor(state_block_indices, dtype=torch.long)
    state_valid_tensor = torch.tensor(state_valid_mask, dtype=torch.bool)

    if state_block_table is None:
        state_block_ids = torch.full_like(state_block_index_tensor, -1, dtype=torch.int32)
    else:
        state_block_index_device = state_block_index_tensor.to(device=state_block_table.device)
        state_valid_device = state_valid_tensor.to(device=state_block_table.device)
        request_indices = torch.arange(len(state_block_indices), dtype=torch.long, device=state_block_table.device)
        safe_block_indices = torch.clamp(state_block_index_device, min=0, max=max(max_state_blocks - 1, 0))
        state_block_ids = state_block_table[request_indices, safe_block_indices].to(torch.int32)
        state_block_ids = torch.where(
            state_valid_device,
            state_block_ids,
            torch.full_like(state_block_ids, -1),
        )
        state_block_ids = state_block_ids.cpu()

    return DSACPStateBroadcastPlan(
        source_ranks=source_rank_tensor,
        local_request_indices=local_request_tensor,
        tail_token_offsets=tail_offset_tensor,
        state_block_ids=state_block_ids,
        state_block_indices=state_block_index_tensor,
        state_valid_mask=state_valid_tensor,
    )


def build_dsa_cp_local_compressed_range(
    input_positions: torch.Tensor,
    compress_ratio: int,
    local_cache_plan: DSACPLocalCachePlan,
    num_actual_tokens: int,
) -> tuple[int, int]:
    if compress_ratio <= 1:
        return 0, 0

    actual_positions = input_positions[:num_actual_tokens]
    compressed_mask = ((actual_positions + 1) % compress_ratio) == 0
    token_offsets = torch.arange(num_actual_tokens, device=actual_positions.device)
    local_mask = torch.zeros_like(compressed_mask, dtype=torch.bool)
    for valid_start, valid_end in local_cache_plan.local_valid_ranges:
        valid_start = min(valid_start, num_actual_tokens)
        valid_end = min(valid_end, num_actual_tokens)
        if valid_start < valid_end:
            local_mask |= (token_offsets >= valid_start) & (token_offsets < valid_end)

    local_compressed = compressed_mask & local_mask
    if not local_compressed.any():
        compressed_count = int(compressed_mask.sum().item())
        if not local_cache_plan.local_valid_ranges:
            return compressed_count, compressed_count
        first_local_start = min(start for start, _ in local_cache_plan.local_valid_ranges)
        before_local = compressed_mask & (token_offsets < first_local_start)
        compressed_start = int(before_local.sum().item())
        return compressed_start, compressed_start

    compressed_indices = torch.cumsum(compressed_mask.to(torch.long), dim=0) - 1
    selected_indices = compressed_indices[local_compressed]
    return int(selected_indices[0].item()), int(selected_indices[-1].item()) + 1


def build_dsa_cp_local_compressor_slot_plan(
    input_positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    local_cache_plan: DSACPLocalCachePlan,
    query_start_loc: list[int],
    num_actual_tokens: int,
    overlap_tokens: int,
    request_ids: list[str] | None = None,
) -> DSACPCompressorSlotPlan:
    slot_shape = tuple(slot_mapping.shape[1:])
    if compress_ratio <= 1:
        empty_slot_mapping = slot_mapping.new_empty((0, *slot_shape))
        empty_bool = torch.empty((0,), dtype=torch.bool, device=slot_mapping.device)
        empty_indices = torch.empty((0,), dtype=torch.long, device=slot_mapping.device)
        empty_start_loc = torch.zeros((1,), dtype=torch.int32, device=slot_mapping.device)
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=slot_mapping.device)
        return DSACPCompressorSlotPlan(
            input_ranges=[],
            valid_ranges=[],
            overlap_ranges=[],
            slot_mapping=empty_slot_mapping,
            valid_output_mask=empty_bool,
            output_indices=empty_indices,
            compressed_positions=empty_indices,
            input_indices=empty_indices,
            input_query_start_loc=empty_start_loc,
            request_indices=empty_indices,
            request_indices_cpu=torch.empty((0,), dtype=torch.long),
            request_ids=None,
            start_pos_offsets=empty_i32,
            prefix_lengths=empty_i32,
            current_start_positions=empty_i32,
            has_prefix_hidden=False,
            all_rank_valid_output_counts=(),
            all_rank_slot_mappings=(),
        )
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")

    actual_positions = input_positions[:num_actual_tokens]
    compressed_mask = ((actual_positions + 1) % compress_ratio) == 0
    compressed_indices = torch.cumsum(compressed_mask.to(torch.long), dim=0) - 1
    invalid_slot = slot_mapping.new_full((1, *slot_shape), -1)
    output_slots: list[torch.Tensor] = []
    output_valid_masks: list[torch.Tensor] = []
    output_indices: list[torch.Tensor] = []
    compressed_positions: list[torch.Tensor] = []
    input_indices: list[torch.Tensor] = []
    input_lengths: list[int] = []
    request_indices: list[int] = []
    start_pos_offsets: list[int] = []
    prefix_lengths: list[int] = []
    current_start_positions: list[int] = []
    input_ranges: list[tuple[int, int]] = []
    valid_ranges: list[tuple[int, int]] = []
    overlap_ranges: list[tuple[int, int]] = []

    token_offsets = torch.arange(num_actual_tokens, device=actual_positions.device)
    all_rank_valid_output_counts = []
    all_rank_slot_mappings = []
    for rank_ranges in local_cache_plan.rank_valid_ranges:
        rank_mask = torch.zeros_like(compressed_mask, dtype=torch.bool)
        for rank_start, rank_end in rank_ranges:
            rank_start = min(rank_start, num_actual_tokens)
            rank_end = min(rank_end, num_actual_tokens)
            if rank_start < rank_end:
                rank_mask |= (token_offsets >= rank_start) & (token_offsets < rank_end)
        rank_mask &= compressed_mask
        rank_count = int(rank_mask.sum().item())
        all_rank_valid_output_counts.append(rank_count)
        if rank_count > 0:
            rank_full_indices = compressed_indices[rank_mask].to(device=slot_mapping.device, dtype=torch.long)
            all_rank_slot_mappings.append(slot_mapping[rank_full_indices])
        else:
            all_rank_slot_mappings.append(slot_mapping.new_empty((0, *slot_shape)))

    for req_idx, (req_start, req_end) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
        req_start = min(req_start, num_actual_tokens)
        req_end = min(req_end, num_actual_tokens)
        if req_idx >= len(local_cache_plan.rank_valid_ranges[local_cache_plan.cp_rank]):
            continue
        valid_start, valid_end = local_cache_plan.rank_valid_ranges[local_cache_plan.cp_rank][req_idx]
        valid_start = max(req_start, min(valid_start, num_actual_tokens))
        valid_end = min(req_end, min(valid_end, num_actual_tokens))
        if valid_start >= valid_end:
            continue

        current_start_pos = int(actual_positions[req_start].item()) if req_start < req_end else 0
        current_valid_start_pos = int(actual_positions[valid_start].item())
        current_valid_end_pos = int(actual_positions[valid_end - 1].item()) + 1
        owner_offsets = torch.nonzero(
            compressed_mask[valid_start:valid_end], as_tuple=False
        ).flatten() + valid_start
        if owner_offsets.numel() > 0:
            first_owner_pos = int(actual_positions[owner_offsets[0]].item())
            needed_abs_start = max(0, first_owner_pos + 1 - compress_ratio)
        else:
            needed_abs_start = current_valid_start_pos
        input_start = max(req_start, valid_start - overlap_tokens)
        input_start_pos = int(actual_positions[input_start].item())
        input_abs_start = min(input_start_pos, needed_abs_start)
        prefix_len = max(0, current_start_pos - input_abs_start)
        if prefix_len > 0:
            input_start = req_start
        input_end = valid_end
        input_ranges.append((input_start, input_end))
        valid_ranges.append((valid_start, valid_end))
        overlap_ranges.append((input_start, valid_start))
        input_indices.append(torch.arange(input_start, input_end, dtype=torch.long, device=slot_mapping.device))
        input_lengths.append(prefix_len + input_end - input_start)
        request_indices.append(req_idx)
        start_pos_offsets.append(input_abs_start - current_start_pos)
        prefix_lengths.append(prefix_len)
        current_start_positions.append(current_start_pos)

        output_positions = torch.arange(
            input_abs_start,
            current_valid_end_pos,
            dtype=actual_positions.dtype,
            device=actual_positions.device,
        )
        local_compressed_positions_mask = ((output_positions + 1) % compress_ratio) == 0
        local_output_positions = output_positions[local_compressed_positions_mask]
        if local_output_positions.numel() == 0:
            continue

        owner_mask = (local_output_positions >= current_valid_start_pos) & (
            local_output_positions < current_valid_end_pos
        )
        local_slots = invalid_slot.expand(local_output_positions.numel(), *slot_shape).clone()
        full_indices = slot_mapping.new_full((local_output_positions.numel(),), -1, dtype=torch.long)
        output_offsets_in_current = local_output_positions - current_start_pos + req_start
        output_offsets_in_current = output_offsets_in_current.to(dtype=torch.long, device=compressed_indices.device)
        in_current_chunk = (output_offsets_in_current >= req_start) & (output_offsets_in_current < req_end)
        if in_current_chunk.any():
            in_current_device = in_current_chunk.to(device=slot_mapping.device)
            known_full_indices = compressed_indices[output_offsets_in_current[in_current_chunk]].to(
                device=slot_mapping.device
            )
            full_indices[in_current_device] = known_full_indices
        if owner_mask.any():
            owner_mask_device = owner_mask.to(device=slot_mapping.device)
            owner_full_indices = full_indices[owner_mask_device]
            local_slots[owner_mask_device] = slot_mapping[owner_full_indices]
        owner_mask_device = owner_mask.to(device=slot_mapping.device)
        output_slots.append(local_slots)
        output_valid_masks.append(owner_mask_device)
        output_indices.append(full_indices)
        compressed_positions.append((local_output_positions + 1 - compress_ratio).to(device=slot_mapping.device))

    if output_slots:
        local_slot_mapping = torch.cat(output_slots)
        valid_output_mask = torch.cat(output_valid_masks)
        local_output_indices = torch.cat(output_indices)
        local_compressed_positions = torch.cat(compressed_positions)
    else:
        local_slot_mapping = slot_mapping.new_empty((0, *slot_shape))
        valid_output_mask = torch.empty((0,), dtype=torch.bool, device=slot_mapping.device)
        local_output_indices = slot_mapping.new_empty((0,), dtype=torch.long)
        local_compressed_positions = slot_mapping.new_empty((0,), dtype=input_positions.dtype)

    if input_indices:
        local_input_indices = torch.cat(input_indices)
        local_input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device=slot_mapping.device)
        local_input_query_start_loc = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=slot_mapping.device),
                torch.cumsum(local_input_lengths, dim=0),
            ]
        )
        local_request_indices = torch.tensor(request_indices, dtype=torch.long, device=slot_mapping.device)
        local_request_indices_cpu = torch.tensor(request_indices, dtype=torch.long)
        local_start_pos_offsets = torch.tensor(start_pos_offsets, dtype=torch.int32, device=slot_mapping.device)
        local_prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.int32)
        local_current_start_positions = torch.tensor(
            current_start_positions, dtype=torch.int32
        )
    else:
        local_input_indices = slot_mapping.new_empty((0,), dtype=torch.long)
        local_input_query_start_loc = torch.zeros((1,), dtype=torch.int32, device=slot_mapping.device)
        local_request_indices = slot_mapping.new_empty((0,), dtype=torch.long)
        local_request_indices_cpu = torch.empty((0,), dtype=torch.long)
        local_start_pos_offsets = torch.empty((0,), dtype=torch.int32, device=slot_mapping.device)
        local_prefix_lengths = torch.empty((0,), dtype=torch.int32)
        local_current_start_positions = torch.empty((0,), dtype=torch.int32)

    return DSACPCompressorSlotPlan(
        input_ranges=input_ranges,
        valid_ranges=valid_ranges,
        overlap_ranges=overlap_ranges,
        slot_mapping=local_slot_mapping,
        valid_output_mask=valid_output_mask,
        output_indices=local_output_indices,
        compressed_positions=local_compressed_positions,
        input_indices=local_input_indices,
        input_query_start_loc=local_input_query_start_loc,
        request_indices=local_request_indices,
        request_indices_cpu=local_request_indices_cpu,
        request_ids=request_ids,
        start_pos_offsets=local_start_pos_offsets,
        prefix_lengths=local_prefix_lengths,
        current_start_positions=local_current_start_positions,
        has_prefix_hidden=any(prefix_len > 0 for prefix_len in prefix_lengths),
        all_rank_valid_output_counts=tuple(all_rank_valid_output_counts),
        all_rank_slot_mappings=tuple(all_rank_slot_mappings),
    )


def build_dsa_cp_swa_window_plan(
    local_cache_plan: DSACPLocalCachePlan,
    query_start_loc: list[int],
    num_actual_tokens: int,
    slot_mapping: torch.Tensor | None = None,
    halo_size: int = DSACP_LOCAL_CACHE_UNIT_SIZE,
) -> DSACPSWAWindowPlan:
    if halo_size < 0:
        raise ValueError(f"halo_size must be non-negative, got {halo_size}")

    input_ranges: list[tuple[int, int]] = []
    valid_ranges: list[tuple[int, int]] = []
    halo_ranges: list[tuple[int, int]] = []
    all_rank_valid_token_counts = get_dsa_cp_all_rank_token_counts(local_cache_plan, num_actual_tokens)
    if slot_mapping is None:
        slot_mapping = torch.arange(num_actual_tokens, dtype=torch.long)
    slot_shape = tuple(slot_mapping.shape[1:])
    all_rank_slot_mappings = []
    for rank_ranges in local_cache_plan.rank_valid_ranges:
        rank_slots = []
        for rank_start, rank_end in rank_ranges:
            rank_start = min(rank_start, num_actual_tokens)
            rank_end = min(rank_end, num_actual_tokens)
            if rank_start < rank_end:
                rank_slots.append(slot_mapping[rank_start:rank_end])
        if rank_slots:
            all_rank_slot_mappings.append(torch.cat(rank_slots, dim=0))
        else:
            all_rank_slot_mappings.append(slot_mapping.new_empty((0, *slot_shape)))

    local_rank_ranges = local_cache_plan.rank_valid_ranges[local_cache_plan.cp_rank]
    for req_idx, (req_start, req_end) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
        req_start = min(req_start, num_actual_tokens)
        req_end = min(req_end, num_actual_tokens)
        if req_idx >= len(local_rank_ranges):
            continue
        valid_start, valid_end = local_rank_ranges[req_idx]
        valid_start = max(req_start, min(valid_start, num_actual_tokens))
        valid_end = min(req_end, min(valid_end, num_actual_tokens))
        if valid_start >= valid_end:
            continue

        halo_start = max(req_start, valid_start - halo_size)
        input_ranges.append((halo_start, valid_end))
        valid_ranges.append((valid_start, valid_end))
        halo_ranges.append((halo_start, valid_start))

    return DSACPSWAWindowPlan(
        input_ranges=input_ranges,
        valid_ranges=valid_ranges,
        halo_ranges=halo_ranges,
        all_rank_valid_token_counts=all_rank_valid_token_counts,
        all_rank_slot_mappings=tuple(all_rank_slot_mappings),
    )


def build_dsa_cp_local_cache_plan(
    num_input_tokens: int,
    cp_size: int,
    cp_rank: int,
    unit_size: int = DSACP_LOCAL_CACHE_UNIT_SIZE,
    query_start_loc: list[int] | tuple[int, ...] | None = None,
) -> DSACPLocalCachePlan:
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if unit_size <= 0:
        raise ValueError(f"unit_size must be positive, got {unit_size}")

    if query_start_loc is None:
        query_start_loc_tuple = (0, num_input_tokens)
    else:
        query_start_loc_tuple = tuple(int(v) for v in query_start_loc)
        if len(query_start_loc_tuple) == 0:
            query_start_loc_tuple = (0, num_input_tokens)
        if query_start_loc_tuple[0] != 0:
            raise ValueError("query_start_loc must start from 0")

    request_spans: list[tuple[int, int, int, int]] = []
    padded_total = 0
    for req_start, req_end in zip(query_start_loc_tuple[:-1], query_start_loc_tuple[1:]):
        req_start = min(req_start, num_input_tokens)
        req_end = min(req_end, num_input_tokens)
        req_len = max(0, req_end - req_start)
        padded_len = _ceil_to_unit(req_len, unit_size)
        request_spans.append((req_start, req_end, padded_total, padded_total + padded_len))
        padded_total += padded_len

    rank_request_ranges: list[tuple[tuple[int, int], ...]] = []
    rank_valid_ranges: list[tuple[tuple[int, int], ...]] = []
    for rank in range(cp_size):
        rank_padded_start, rank_padded_end = _get_dsa_cp_local_range(
            padded_total,
            cp_size,
            rank,
            unit_size,
        )
        request_ranges: list[tuple[int, int]] = []
        valid_ranges: list[tuple[int, int]] = []
        for req_start, req_end, req_padded_start, req_padded_end in request_spans:
            req_len = max(0, req_end - req_start)
            owner_start = max(rank_padded_start, req_padded_start)
            owner_end = min(rank_padded_end, req_padded_end)
            if owner_start >= owner_end:
                empty_pos = req_start if rank_padded_end <= req_padded_start else req_end
                request_ranges.append((empty_pos, empty_pos))
                valid_ranges.append((empty_pos, empty_pos))
                continue

            real_start = _logical_padded_offset_to_real(
                owner_start,
                req_padded_start,
                req_start,
                req_len,
            )
            real_end = _logical_padded_offset_to_real(
                owner_end,
                req_padded_start,
                req_start,
                req_len,
            )
            if real_start >= real_end:
                real_start = real_end
            request_ranges.append((real_start, real_end))
            valid_ranges.append((real_start, real_end))
        rank_request_ranges.append(tuple(request_ranges))
        rank_valid_ranges.append(tuple(valid_ranges))

    local_request_ranges = rank_request_ranges[cp_rank]
    local_valid_ranges = _filter_non_empty_ranges(rank_valid_ranges[cp_rank])
    local_offsets = _build_local_offsets(local_valid_ranges)
    all_rank_num_tokens = tuple(_sum_ranges(_filter_non_empty_ranges(ranges)) for ranges in rank_valid_ranges)
    local_num_tokens = all_rank_num_tokens[cp_rank]
    local_start = min((start for start, _ in local_valid_ranges), default=0)
    local_end = max((end for _, end in local_valid_ranges), default=local_start)

    return DSACPLocalCachePlan(
        enabled=True,
        cp_size=cp_size,
        cp_rank=cp_rank,
        query_start_loc=query_start_loc_tuple,
        rank_request_ranges=tuple(rank_request_ranges),
        rank_valid_ranges=tuple(rank_valid_ranges),
        local_request_ranges=local_request_ranges,
        local_valid_ranges=local_valid_ranges,
        local_offsets=local_offsets,
        all_rank_num_tokens=all_rank_num_tokens,
        local_start=local_start,
        local_end=local_end,
        tokens_per_rank=local_num_tokens,
        num_tokens_pad=padded_total,
        unit_size=unit_size,
        local_num_tokens=local_num_tokens,
    )


@dataclass
class DSACPMetadata:
    """Context-parallel metadata for sequence-sharded DSA execution."""

    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    local_start: int
    local_end: int
    tokens_per_rank: int
    num_tokens_pad: int
    local_sin: torch.Tensor = None
    local_cos: torch.Tensor = None
    local_cache_plan: DSACPLocalCachePlan | None = None
    swa_window_plan: DSACPSWAWindowPlan | None = None
    swa_hidden_input_plan: DSACPHiddenInputPlan | None = None
    swa_slot_mapping: torch.Tensor | None = None
    swa_valid_start: int = 0
    swa_valid_end: int = 0
    compressor_slot_plan: DSACPCompressorSlotPlan | None = None
    compressor_hidden_input_plan: DSACPHiddenInputPlan | None = None
    state_broadcast_plan: DSACPStateBroadcastPlan | None = None
    compressed_slot_mapping: torch.Tensor | None = None
    compressed_valid_start: int = 0
    compressed_valid_end: int = 0


@dataclass
class AscendDSAReqMetadata:
    """Unified per-request metadata — combines fields formerly split into
    prefill and decode sub-structures.

    All methods (builder, forward) operate on this single metadata,
    without distinguishing prefill vs decode request types.
    """

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None
    block_size: int
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    cp_metadata: DSACPMetadata
    input_positions_cpu: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    num_compressed_tokens: int | None = None
    request_ids: list[str] | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor = None
    num_reqs_actual: int | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_cmp_seqlen_list: torch.Tensor = None
    attn_mask: torch.Tensor | None = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: AscendDSAReqMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None

    start_pos: torch.Tensor | None = None


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSACPMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: torch.Tensor | None = None
    req_sas_metadata: torch.Tensor
    req_qli_metadata: torch.Tensor
    block_size: int = 128
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config

        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim

        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.seq_lens_cpu: torch.Tensor = None

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        self.enable_dsa_cp_local_cache = ascend_envs.VLLM_ASCEND_ENABLE_DSA_CP_LOCAL_CACHE
        try:
            tp_group = get_tp_group()
            self.cp_size = tp_group.world_size
            self.cp_rank = tp_group.rank_in_group
        except Exception:
            self.cp_size = 1
            self.cp_rank = 0
        hf_config = self.model_config.hf_config

        if AscendDSACPMetadataBuilder.hadamard is None:
            if hf_config.model_type == "deepseek_v4":
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard  # type: ignore[import-untyped]
                except ImportError as e:
                    raise ImportError(
                        "DeepSeek-V4 indexer attention requires SciPy for Hadamard transform. Please install scipy."
                    ) from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                if self.vllm_config.model_config.enable_sleep_mode:
                    # Sleep mode allocates KV inside CaMemAllocator; tag Hadamard so
                    # sleep/wake does not treat it as KV cache.
                    from vllm_ascend.device_allocator.camem import CaMemAllocator

                    allocator = CaMemAllocator.get_instance()
                    with allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag):
                        AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                            hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                        ).to(torch.bfloat16)
                else:
                    AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                        hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                    ).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.req_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.req_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        self.local_query_start_loc = torch.zeros(
            scheduler_config.max_num_seqs + 1, dtype=torch.int32, device=self.device
        )
        self.local_seq_lens = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_local_query_start_loc = [
                torch.zeros(scheduler_config.max_num_seqs + 1, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_local_seq_lens = [
                torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSACPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual")
        self.block_size = kwargs.get("block_size", 128)

        common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert common_ratio_to_sas_metadata is not None
        self.common_ratio_to_sas_metadata = common_ratio_to_sas_metadata
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        attn_state = kwargs.get("attn_state", common_attn_metadata.attn_state)
        has_prefill = _has_prefill(attn_state)

        num_input_tokens = common_attn_metadata.num_input_tokens
        if self.common_ratio_to_sas_metadata.get("input_positions", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.decode_threshold,
                    treat_short_extends_as_decodes=False,
                )
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            input_positions_cpu = common_attn_metadata.positions_cpu[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            self.common_ratio_to_sas_metadata["input_positions_cpu"] = input_positions_cpu
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            self.seq_lens_cpu = _seq_lens_cpu
            self.common_ratio_to_sas_metadata["seq_lens_cpu"] = self.seq_lens_cpu
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            input_positions_cpu = self.common_ratio_to_sas_metadata["input_positions_cpu"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        req_metadata = self.build_req_metadata(
            common_attn_metadata, input_positions, input_positions_cpu, num_input_tokens, num_reqs_actual, attn_state
        )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            attn_mask=None,
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=attn_state,
            req_metadata=req_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSACPMetadataBuilder.hadamard,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )

        self.num_decodes = num_decodes
        self.num_prefills = num_prefills
        self.num_decode_tokens = num_decode_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        if common_attn_metadata._seq_lens_cpu is not None:
            self.seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            self.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            self.seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        self.block_size = kwargs.get("block_size", 128)

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        input_positions_cpu = common_attn_metadata.positions_cpu[:num_input_tokens].long()
        # Draft steps update positions independently. Reusing the global RoPE
        # cache can let later draft steps overwrite step-0 metadata.
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        formatted_slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = formatted_slot_mapping
        self.slot_mapping[:num_input_tokens] = formatted_slot_mapping

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        req_metadata = self.build_req_metadata_for_drafting(
            draft_index=draft_index,
            common_attn_metadata=common_attn_metadata,
            input_positions=input_positions,
            input_positions_cpu=input_positions_cpu,
            num_input_tokens=num_input_tokens,
        )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            attn_mask=None,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            query_start_loc=common_attn_metadata.query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=None,
        )

    def build_req_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor,
        input_positions_cpu: torch.Tensor,
        num_input_tokens: int,
    ) -> AscendDSAReqMetadata:
        """Build DSA-CP metadata for one draft step."""
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        has_prefill = _has_prefill(common_attn_metadata.attn_state)

        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)
        local_cache_plan = (
            self._build_local_cache_plan(
                num_input_tokens, query_start_loc_cpu[: num_reqs + 1].tolist()
            )
            if has_prefill
            else None
        )
        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
            local_cos,
            local_sin,
        ) = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            use_cache=False,
            local_query_start_loc=self.spec_local_query_start_loc[draft_index - 1],
            local_seq_lens=self.spec_local_seq_lens[draft_index - 1],
            local_cache_plan=local_cache_plan,
        )
        local_query_start_loc = local_query_start_loc.clone()
        local_seq_lens = local_seq_lens.clone()

        _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu, _, _ = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=None,
            query_start_loc=query_start_loc_cpu,
            seq_lens=self.seq_lens_cpu[:num_reqs],
            use_cache=False,
            local_cache_plan=local_cache_plan,
        )
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))

        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_index - 1][: self.num_actual_tokens]
        actual_num_tokens = self.num_actual_tokens
        swa_slot_mapping, swa_valid_start, swa_valid_end = self._build_swa_local_slot_mapping(
            local_cache_plan, actual_num_tokens
        )
        swa_window_plan = self._build_swa_window_plan(
            local_cache_plan, query_start_loc_cpu[: num_reqs + 1].tolist(), actual_num_tokens
        )
        swa_hidden_input_plan = (
            build_dsa_cp_hidden_input_plan(
                input_ranges=swa_window_plan.input_ranges,
                local_cache_plan=local_cache_plan,
                num_actual_tokens=actual_num_tokens,
            )
            if local_cache_plan is not None and swa_window_plan is not None
            else None
        )

        num_heads = self.model_config.hf_config.num_attention_heads
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        metadata_kwargs.setdefault("device", str(self.seqused_q.device))
        cu_seqlens_ori_kv = (
            local_query_start_loc
            if has_prefill
            else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                None,
                "draft_cu_seqlens_ori_kv",
                local_seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
        )
        cu_seqlens_cmp_kv = (
            None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        )
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=num_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=local_query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=local_seq_lens,
            max_seqlen_q=max_local_query_len,
            max_seqlen_kv=max_local_seq_lens,
            batch_size=num_reqs,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        cp_metadata = DSACPMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seq_lens=local_seq_lens,
            local_start=local_start,
            local_end=local_end_with_pad,
            tokens_per_rank=tokens_per_rank,
            num_tokens_pad=num_tokens_pad,
            local_sin=local_sin,
            local_cos=local_cos,
            local_cache_plan=local_cache_plan,
            swa_window_plan=swa_window_plan,
            swa_hidden_input_plan=swa_hidden_input_plan,
            swa_slot_mapping=swa_slot_mapping,
            swa_valid_start=swa_valid_start,
            swa_valid_end=swa_valid_end,
        )

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            input_positions_cpu=input_positions_cpu[: self.num_actual_tokens] if has_prefill else None,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu[: num_reqs + 1] if has_prefill else None,
            cp_metadata=cp_metadata,
            request_ids=common_attn_metadata.request_ids[:num_reqs]
            if common_attn_metadata.request_ids is not None
            else None,
            sin=sin,
            cos=cos,
            start_pos=start_pos,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_cmp_seqlen_list=None,
        )

    def _num_compressor_metadata_rows(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + common_attn_metadata.num_reqs)

    def _get_slot_mapping_size(
        self,
        input_positions_cpu: torch.Tensor,
        compress_ratio: int,
        num_reqs: int,
        num_actual_tokens: int,
    ) -> int:
        if compress_ratio <= 1:
            return num_actual_tokens
        # Compressor metadata produces at most one compressed row per ratio
        # group plus one boundary row per request, capped by valid tokens.
        return min(num_actual_tokens, num_actual_tokens // compress_ratio + num_reqs)

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor,
        input_positions_cpu: torch.Tensor,
        num_input_tokens: int,
        num_reqs_actual: int | None,
        attn_state: AscendAttentionState,
    ) -> AscendDSAReqMetadata:
        """Build a single unified metadata for all requests (prefill + decode)."""
        num_reqs = common_attn_metadata.num_reqs
        has_prefill = _has_prefill(attn_state)
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]

        # cos/sin for all tokens
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)

        local_cache_plan = (
            self._build_local_cache_plan(
                num_input_tokens, query_start_loc_cpu[: num_reqs + 1].tolist()
            )
            if has_prefill
            else None
        )
        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
            local_cos,
            local_sin,
        ) = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            use_cache=not has_prefill,
            local_query_start_loc=self.local_query_start_loc,
            local_seq_lens=self.local_seq_lens,
            local_cache_plan=local_cache_plan,
        )
        local_seq_lens_q = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]

        _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu, _, _ = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=None,
            query_start_loc=query_start_loc_cpu,
            seq_lens=self.seq_lens_cpu[:num_reqs],
            use_cache=False,
            local_cache_plan=local_cache_plan,
        )
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))

        # start_pos: context length before current query
        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = start_pos

        if num_reqs_actual is None:
            num_reqs_actual = num_reqs
        else:
            num_reqs_actual = min(num_reqs_actual, num_reqs)
            if num_reqs_actual < num_reqs:
                self.start_pos_prefill[num_reqs_actual:].fill_(0)
                self.block_table[num_reqs_actual:num_reqs, ...].fill_(0)

        # --- Local SWA/window cache owner mapping ---
        actual_num_tokens = self.num_actual_tokens
        swa_slot_mapping, swa_valid_start, swa_valid_end = self._build_swa_local_slot_mapping(
            local_cache_plan, actual_num_tokens
        )
        swa_window_plan = self._build_swa_window_plan(
            local_cache_plan, query_start_loc_cpu[: num_reqs + 1].tolist(), actual_num_tokens
        )

        # --- Compressed positions ---
        full_compress_cos, full_compress_sin = None, None
        num_compressed_tokens = None
        compressed_slot_mapping = None
        compressed_valid_start = 0
        compressed_valid_end = 0
        compressor_slot_plan = None
        state_broadcast_plan = None
        swa_hidden_input_plan = None
        compressor_hidden_input_plan = None
        cu_cmp_seqlens = self._get_cmp_seqlens_for_metadata(has_prefill)
        actual_input_positions_cpu = input_positions_cpu[:actual_num_tokens]
        slot_mapping_size = self._get_slot_mapping_size(
            actual_input_positions_cpu, self.compressor_ratio, num_reqs, actual_num_tokens
        )
        slot_mapping = self.slot_mapping[:slot_mapping_size]
        (
            compressed_slot_mapping,
            compressed_valid_start,
            compressed_valid_end,
        ) = self._build_compressed_local_slot_mapping(
            local_cache_plan=local_cache_plan,
            input_positions=actual_input_positions_cpu,
            num_actual_tokens=actual_num_tokens,
            compress_ratio=self.compressor_ratio,
        )
        compressor_slot_plan = self._build_local_compressor_slot_plan(
            local_cache_plan=local_cache_plan,
            input_positions=actual_input_positions_cpu,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc_cpu[: num_reqs + 1].tolist(),
            num_actual_tokens=actual_num_tokens,
            compress_ratio=self.compressor_ratio,
            request_ids=common_attn_metadata.request_ids[:num_reqs]
            if common_attn_metadata.request_ids is not None
            else None,
        )
        state_broadcast_plan = self._build_state_broadcast_plan(
            local_cache_plan=local_cache_plan,
            query_start_loc=query_start_loc_cpu[: num_reqs + 1].tolist(),
            num_actual_tokens=actual_num_tokens,
            input_positions=actual_input_positions_cpu,
        )
        swa_hidden_input_plan = (
            build_dsa_cp_hidden_input_plan(
                input_ranges=swa_window_plan.input_ranges,
                local_cache_plan=local_cache_plan,
                num_actual_tokens=actual_num_tokens,
            )
            if local_cache_plan is not None and swa_window_plan is not None
            else None
        )
        compressor_hidden_input_plan = (
            build_dsa_cp_hidden_input_plan(
                input_ranges=compressor_slot_plan.input_ranges,
                local_cache_plan=local_cache_plan,
                num_actual_tokens=actual_num_tokens,
            )
            if local_cache_plan is not None and compressor_slot_plan is not None
            else None
        )

        if self.compressor_ratio > 1:
            layer_name = f"c{self.compressor_ratio}"
            # Keep graph inputs here. The actual compressor slot mapping is
            # produced by the metadata op in forward unless local-cache CP
            # supplies an owner-only slot plan.
            num_compressed_tokens = self._num_compressor_metadata_rows(common_attn_metadata)
            full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
            if compressor_slot_plan is None:
                slot_mapping = None

        # --- SAS metadata (all requests combined) ---
        num_heads = self.model_config.hf_config.num_attention_heads
        index_topk = self.model_config.hf_config.index_topk

        sas_metadata = self._build_sas_metadata(
            num_heads=num_heads,
            query_start_loc=local_query_start_loc,
            seq_lens=local_seq_lens,
            seq_lens_q=local_seq_lens_q,
            max_query_len=max_local_query_len,
            max_seq_lens=max_local_seq_lens,
            index_topk=index_topk,
            num_reqs=num_reqs,
            has_prefill=has_prefill,
            cu_cmp_seqlen_list=cu_cmp_seqlens,
        )

        # --- QLI metadata (all requests combined) ---
        qli_metadata = self._build_qli_metadata(
            query_start_loc=local_query_start_loc,
            seq_lens=local_seq_lens,
            seq_lens_q=local_seq_lens_q,
            num_reqs=num_reqs,
        )

        cp_metadata = DSACPMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seq_lens=local_seq_lens,
            local_start=local_start,
            local_end=local_end_with_pad,
            tokens_per_rank=tokens_per_rank,
            num_tokens_pad=num_tokens_pad,
            local_sin=local_sin,
            local_cos=local_cos,
            local_cache_plan=local_cache_plan,
            swa_window_plan=swa_window_plan,
            swa_hidden_input_plan=swa_hidden_input_plan,
            swa_slot_mapping=swa_slot_mapping,
            swa_valid_start=swa_valid_start,
            swa_valid_end=swa_valid_end,
            compressor_slot_plan=compressor_slot_plan,
            compressor_hidden_input_plan=compressor_hidden_input_plan,
            state_broadcast_plan=state_broadcast_plan,
            compressed_slot_mapping=compressed_slot_mapping,
            compressed_valid_start=compressed_valid_start,
            compressed_valid_end=compressed_valid_end,
        )

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            input_positions_cpu=input_positions_cpu[: self.num_actual_tokens] if has_prefill else None,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu[: num_reqs + 1] if has_prefill else None,
            cp_metadata=cp_metadata,
            request_ids=common_attn_metadata.request_ids[:num_reqs]
            if common_attn_metadata.request_ids is not None
            else None,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_compressed_tokens=num_compressed_tokens,
            num_reqs_actual=num_reqs_actual,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_cmp_seqlen_list=cu_cmp_seqlens,
        )

    def _build_local_cache_plan(
        self, num_input_tokens: int, query_start_loc: list[int] | None = None
    ) -> DSACPLocalCachePlan | None:
        if not self.enable_dsa_cp_local_cache or self.cp_size <= 1:
            return None

        return build_dsa_cp_local_cache_plan(
            num_input_tokens=num_input_tokens,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            query_start_loc=query_start_loc,
        )

    def _build_swa_local_slot_mapping(
        self,
        local_cache_plan: DSACPLocalCachePlan | None,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor | None, int, int]:
        if local_cache_plan is None:
            return None, 0, 0

        slot_mappings = []
        for valid_start, valid_end in local_cache_plan.local_valid_ranges:
            valid_start = min(valid_start, num_actual_tokens)
            valid_end = min(valid_end, num_actual_tokens)
            if valid_start < valid_end:
                slot_mappings.append(self.slot_mapping[valid_start:valid_end])
        if not slot_mappings:
            return self.slot_mapping[:0], 0, 0
        return torch.cat(slot_mappings, dim=0), 0, sum(
            end - start for start, end in local_cache_plan.local_valid_ranges
        )

    def _build_swa_window_plan(
        self,
        local_cache_plan: DSACPLocalCachePlan | None,
        query_start_loc: list[int],
        num_actual_tokens: int,
    ) -> DSACPSWAWindowPlan | None:
        if local_cache_plan is None:
            return None
        return build_dsa_cp_swa_window_plan(
            local_cache_plan=local_cache_plan,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            slot_mapping=self.slot_mapping,
        )

    def _build_compressed_local_slot_mapping(
        self,
        local_cache_plan: DSACPLocalCachePlan | None,
        input_positions: torch.Tensor,
        num_actual_tokens: int,
        compress_ratio: int,
    ) -> tuple[torch.Tensor | None, int, int]:
        if local_cache_plan is None or compress_ratio <= 1:
            return None, 0, 0

        compressed_start, compressed_end = build_dsa_cp_local_compressed_range(
            input_positions=input_positions,
            compress_ratio=compress_ratio,
            local_cache_plan=local_cache_plan,
            num_actual_tokens=num_actual_tokens,
        )
        return self.slot_mapping[compressed_start:compressed_end], compressed_start, compressed_end

    def _build_local_compressor_slot_plan(
        self,
        local_cache_plan: DSACPLocalCachePlan | None,
        input_positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        query_start_loc: list[int],
        num_actual_tokens: int,
        compress_ratio: int,
        request_ids: list[str] | None = None,
    ) -> DSACPCompressorSlotPlan | None:
        if local_cache_plan is None or compress_ratio <= 1:
            return None

        return build_dsa_cp_local_compressor_slot_plan(
            input_positions=input_positions,
            slot_mapping=slot_mapping,
            compress_ratio=compress_ratio,
            local_cache_plan=local_cache_plan,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            overlap_tokens=compress_ratio,
            request_ids=request_ids,
        )

    def _build_state_broadcast_plan(
        self,
        local_cache_plan: DSACPLocalCachePlan | None,
        query_start_loc: list[int],
        num_actual_tokens: int,
        input_positions: torch.Tensor,
    ) -> DSACPStateBroadcastPlan | None:
        if local_cache_plan is None:
            return None
        return build_dsa_cp_state_broadcast_plan(
            local_cache_plan=local_cache_plan,
            query_start_loc=query_start_loc,
            num_actual_tokens=num_actual_tokens,
            input_positions=input_positions,
            state_block_table=self.block_table,
            compress_ratio=max(1, self.compressor_ratio),
            state_block_size=self.block_size,
        )

    def _build_local_token_metadata(
        self,
        num_reqs,
        num_input_tokens,
        input_positions,
        query_start_loc,
        seq_lens,
        use_cache,
        local_query_start_loc=None,
        local_seq_lens=None,
        local_cache_plan: DSACPLocalCachePlan | None = None,
    ):
        """
        For example:
        If we have TP size 3, num_input_tokens=45, and
        query_start_loc = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45].
        That means we have 9 requests with seq lens [1, 2, 3, 4, 5, 6, 7, 8, 9].
        For tp_rank 1, local_start=15, local_end=30, tokens_per_rank=15.
        local_query_start=[15, 15, 15, 15, 15, 15, 21, 28, 30]
        local_query_end = [15, 15, 15, 15, 15, 21, 28, 30, 30]
        local_query_lens = [0, 0, 0, 0, 0, 6, 7, 2, 0]
        self.local_query_start_loc = [0, 0, 0, 0, 0, 0, 6, 13, 15]
        offset = [-14, -12, -9, -5, 0, 0, 0, 6, 15]
        seq_lens-offset=[15, 14, 12, 9, 5, 6, 7, 2, -6]
        local_reqs_mask = [0, 0, 0, 0, 0, 1, 1, 1, 0]
        local_seq_lens = [0, 0, 0, 0, 0, 6, 7, 2, 0]
        """
        if local_cache_plan is None:
            tp_group = get_tp_group()
            tp_size = tp_group.world_size
            tp_rank = tp_group.rank_in_group
            # Split the flattened token stream evenly across TP ranks. Padding keeps
            # every rank's local slice the same length, which simplifies CP kernels.
            num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
            tokens_per_rank = num_tokens_pad // tp_size
            local_start = tp_rank * tokens_per_rank
            local_end = local_start + tokens_per_rank

            if local_query_start_loc is not None:
                local_query_start_loc.fill_(0)
                local_seq_lens.fill_(0)

            local_query_start = torch.clamp(query_start_loc[:-1], min=local_start, max=local_end)
            local_query_end = torch.clamp(query_start_loc[1:], min=local_start, max=local_end)
            local_query_lens = local_query_end - local_query_start
            if local_query_start_loc is not None:
                local_query_start_loc[1 : num_reqs + 1] = torch.cumsum(local_query_lens, dim=0)
            else:
                local_query_start_loc = torch.cat(
                    [
                        torch.tensor([0], dtype=local_query_lens.dtype, device=local_query_lens.device),
                        torch.cumsum(local_query_lens, dim=0),
                    ],
                    0,
                )

            offset = query_start_loc[1:] - local_query_end
            if local_seq_lens is not None:
                local_seq_lens[:num_reqs] = (local_query_lens > 0) * (seq_lens - offset)
            else:
                local_seq_lens = (local_query_lens > 0) * (seq_lens - offset)

            if input_positions is not None:
                pad_tokens = num_tokens_pad - input_positions.shape[0]
                if pad_tokens > 0:
                    input_positions = F.pad(input_positions, (0, pad_tokens), value=0)
                local_cos, local_sin = get_cos_and_sin_dsa(input_positions, use_cache=use_cache)
                local_cos = local_cos[local_start:local_end]
                local_sin = local_sin[local_start:local_end]
            else:
                local_cos = None
                local_sin = None
        else:
            local_start = local_cache_plan.local_start
            local_end = local_cache_plan.local_end
            tokens_per_rank = local_cache_plan.tokens_per_rank
            num_tokens_pad = local_cache_plan.num_tokens_pad

            if local_query_start_loc is not None:
                local_query_start_loc.fill_(0)
                local_seq_lens.fill_(0)

            query_lens: list[int] = []
            tail_offsets: list[int] = []
            request_starts = local_cache_plan.query_start_loc
            local_rank_ranges = local_cache_plan.rank_valid_ranges[local_cache_plan.cp_rank]
            for req_idx in range(num_reqs):
                req_start = min(int(request_starts[req_idx]), num_input_tokens)
                req_end = min(int(request_starts[req_idx + 1]), num_input_tokens)
                if req_idx >= len(local_rank_ranges):
                    query_lens.append(0)
                    tail_offsets.append(0)
                    continue
                valid_start, valid_end = local_rank_ranges[req_idx]
                valid_start = max(req_start, min(valid_start, num_input_tokens))
                valid_end = min(req_end, min(valid_end, num_input_tokens))
                local_query_len = max(0, valid_end - valid_start)
                query_lens.append(local_query_len)
                tail_offsets.append(req_end - valid_end if local_query_len > 0 else 0)

            local_query_lens = torch.tensor(query_lens, dtype=query_start_loc.dtype, device=query_start_loc.device)
            if local_query_start_loc is not None:
                local_query_start_loc[1 : num_reqs + 1] = torch.cumsum(local_query_lens, dim=0)
            else:
                local_query_start_loc = torch.cat(
                    [
                        torch.tensor([0], dtype=local_query_lens.dtype, device=local_query_lens.device),
                        torch.cumsum(local_query_lens, dim=0),
                    ],
                    0,
                )
            tail_offsets_tensor = torch.tensor(tail_offsets, dtype=seq_lens.dtype, device=seq_lens.device)
            local_query_lens_for_seq = local_query_lens.to(device=seq_lens.device)
            local_seq_lens_tensor = torch.where(
                local_query_lens_for_seq > 0,
                seq_lens[:num_reqs] - tail_offsets_tensor,
                torch.zeros_like(seq_lens[:num_reqs]),
            )
            if local_seq_lens is not None:
                local_seq_lens[:num_reqs] = local_seq_lens_tensor
            else:
                local_seq_lens = local_seq_lens_tensor

            if input_positions is not None:
                cos, sin = get_cos_and_sin_dsa(input_positions[:num_input_tokens], use_cache=use_cache)
                if local_cache_plan.local_valid_ranges:
                    local_cos, local_sin = concatenate_rope_slices(
                        cos,
                        local_cache_plan.local_valid_ranges,
                    )
                else:
                    local_cos = cos[:0]
                    local_sin = sin[:0]
            else:
                local_cos = None
                local_sin = None
        return (
            local_start,
            local_end,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc[: num_reqs + 1],
            local_seq_lens[:num_reqs],
            local_cos,
            local_sin,
        )

    def _get_cmp_seqlens_for_metadata(self, has_prefill):
        if self.compressor_ratio <= 1:
            return None
        if has_prefill:
            return None
        return DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)

    def _build_sas_metadata(
        self,
        num_heads,
        query_start_loc,
        seq_lens,
        seq_lens_q,
        max_query_len,
        max_seq_lens,
        index_topk,
        num_reqs,
        has_prefill,
        cu_cmp_seqlen_list,
    ):
        cmp_ratio = self.compressor_ratio if self.compressor_ratio > 1 else 1
        cache_key = f"cp_sas_c{cmp_ratio}"
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)
        if metadata is None:
            cu_seqlens_ori_kv = (
                query_start_loc
                if has_prefill
                else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                    self.common_ratio_to_sas_metadata,
                    f"{cache_key}_cu_seqlens_ori_kv",
                    seq_lens,
                    num_reqs,
                    self._zero_i32,
                    self.cu_seqlens_ori_kv,
                )
            )
            cu_seqlens_cmp_kv = (
                None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
            )
            metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
            metadata_kwargs.setdefault("device", str(self.seqused_q.device))
            kw = dict(
                **metadata_kwargs,
                num_heads_q=num_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_kv=max_seq_lens,
                batch_size=num_reqs,
                ori_mask_mode=4,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
            )

            if self.compressor_ratio > 1:
                kw["has_cmp_kv"] = True
                if self.compressor_ratio == 4:
                    kw["cmp_mask_mode"] = 3
                    kw["cmp_topk"] = index_topk
                else:
                    kw["cmp_mask_mode"] = 3
                kw["cmp_ratio"] = cmp_ratio
                kw["cu_seqlens_cmp_kv"] = cu_cmp_seqlen_list
            else:
                kw["cmp_ratio"] = cmp_ratio
                kw["has_cmp_kv"] = False

            metadata = metadata_op(**kw)
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        self.req_sas_metadata[:1024] = metadata
        return self.req_sas_metadata[:1024]

    def _build_qli_metadata(self, query_start_loc, seq_lens, seq_lens_q, num_reqs):
        if self.compressor_ratio != 4:
            return None

        cache_key = "cp_qli"
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)

        if metadata is None:
            max_seqlen_q = max(1, int(seq_lens_q.max().item()))
            max_seqlen_k = max(1, int(seq_lens.max().item()))
            metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=seq_lens.clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=num_reqs,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        self.req_qli_metadata[:1024] = metadata
        return self.req_qli_metadata[:1024]

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                attn_state=attn_state,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Graph capture only supports DecodeOnly and SpecDecoding attn states, got {attn_state}."
            )

        assert attn_metadata is not None
        return attn_metadata


class AscendDSACPImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    wo_a_full_pool: ClassVar[torch.Tensor | None] = None
    wo_a_full_weight_scale_pool: ClassVar[torch.Tensor | None] = None
    wo_b_full_pool: ClassVar[torch.Tensor | None] = None
    wo_b_full_weight_scale_pool: ClassVar[torch.Tensor | None] = None

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5
        self.tp_group = get_tp_group()
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs.get("q_norm_without_weight")
        self.kv_norm = kwargs["kv_norm"]

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp() and (
            get_ascend_device_type() == AscendDeviceType.A5
        )
        self._wo_a_dynamic_quant = False
        self._wo_b_dynamic_quant = False

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

        self._dsa_cp_hidden_tail_cache: dict[str, dict[str, tuple[int, torch.Tensor]]] = {}
        self._cached_compressor_kv_trailing: tuple[int, ...] | None = None
        self._cached_indexer_kv_trailing: tuple[int, ...] | None = None

    def _compute_compressor_metadata(
        self,
        metadata: AscendDSAReqMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert metadata.full_compress_cos is not None
        assert metadata.full_compress_sin is not None
        assert metadata.num_compressed_tokens is not None
        assert metadata.start_pos is not None
        assert metadata.num_reqs_actual is not None
        full_compress_cos = metadata.full_compress_cos.view(
            metadata.full_compress_cos.shape[0],
            metadata.full_compress_cos.shape[-1],
        )
        full_compress_sin = metadata.full_compress_sin.view(
            metadata.full_compress_sin.shape[0],
            metadata.full_compress_sin.shape[-1],
        )
        return torch.ops._C_ascend.compressor_metadata(
            full_compress_cos,
            full_compress_sin,
            metadata.query_start_loc,
            metadata.start_pos,
            metadata.block_table,
            metadata.block_size,
            DeviceOperator.get_dsa_compressor_slot_mapping_format(),
            self.compress_ratio,
            metadata.num_compressed_tokens,
            metadata.num_reqs_actual,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.attn_sink.numel() != self.num_heads:
            raise RuntimeError(
                "DSA-CP expects full-head attn_sink loaded on every TP rank, "
                f"got {self.attn_sink.numel()} heads, expected {self.num_heads}."
            )
        if self.enable_dsa_cp_with_o_proj_tp:
            self._maybe_init_o_proj_tp_full_params()

    @staticmethod
    def _check_dynamic_quant(layer: torch.nn.Module) -> bool:
        return get_ascend_device_type() in {AscendDeviceType.A5} and hasattr(layer, "weight_scale")

    def _maybe_init_o_proj_tp_full_params(self) -> None:
        self._wo_a_dynamic_quant = type(self)._check_dynamic_quant(self.wo_a)
        self._wo_b_dynamic_quant = type(self)._check_dynamic_quant(self.wo_b)
        if AscendDSACPImpl.wo_a_full_pool is None:
            sample = self.wo_a.weight
            AscendDSACPImpl.wo_a_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                dtype=sample.dtype,
                device=sample.device,
            )
        self.wo_a_tp_weight = self.wo_a.weight.clone().detach().contiguous()
        self.wo_a.weight.set_(self.wo_a_tp_weight)
        if AscendDSACPImpl.wo_b_full_pool is None:
            sample = self.wo_b.weight
            AscendDSACPImpl.wo_b_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                dtype=sample.dtype,
                device=sample.device,
            )
        self.wo_b_tp_weight = self.wo_b.weight.clone().detach().contiguous()
        self.wo_b.weight.set_(self.wo_b_tp_weight)

        if self._wo_a_dynamic_quant:
            if AscendDSACPImpl.wo_a_full_weight_scale_pool is None:
                sample = self.wo_a.weight_scale
                AscendDSACPImpl.wo_a_full_weight_scale_pool = torch.empty(
                    (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                    dtype=sample.dtype,
                    device=sample.device,
                )
            self.wo_a_tp_weight_scale = self.wo_a.weight_scale.clone().detach().contiguous()
            self.wo_a.weight_scale.set_(self.wo_a_tp_weight_scale)
        if self._wo_b_dynamic_quant:
            if AscendDSACPImpl.wo_b_full_weight_scale_pool is None:
                sample = self.wo_b.weight_scale
                AscendDSACPImpl.wo_b_full_weight_scale_pool = torch.empty(
                    (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                    dtype=sample.dtype,
                    device=sample.device,
                )
            self.wo_b_tp_weight_scale = self.wo_b.weight_scale.clone().detach().contiguous()
            self.wo_b.weight_scale.set_(self.wo_b_tp_weight_scale)

    def _maybe_all_gather_o_proj_full_weight(
        self,
        enabled: bool,
    ) -> list[torch.distributed.Work]:
        if not enabled:
            return []
        handles = []
        assert AscendDSACPImpl.wo_a_full_pool is not None
        _, weight_handle = all_gather_async(
            self.wo_a_tp_weight,
            self.tp_group,
            output=AscendDSACPImpl.wo_a_full_pool,
        )
        if weight_handle is not None:
            handles.append(weight_handle)
        assert AscendDSACPImpl.wo_b_full_pool is not None
        _, wo_b_weight_handle = all_gather_async(
            self.wo_b_tp_weight,
            self.tp_group,
            output=AscendDSACPImpl.wo_b_full_pool,
        )
        if wo_b_weight_handle is not None:
            handles.append(wo_b_weight_handle)
        if self._wo_a_dynamic_quant:
            assert AscendDSACPImpl.wo_a_full_weight_scale_pool is not None
            _, weight_scale_handle = all_gather_async(
                self.wo_a_tp_weight_scale,
                self.tp_group,
                output=AscendDSACPImpl.wo_a_full_weight_scale_pool,
            )
            if weight_scale_handle is not None:
                handles.append(weight_scale_handle)
        if self._wo_b_dynamic_quant:
            assert AscendDSACPImpl.wo_b_full_weight_scale_pool is not None
            _, wo_b_weight_scale_handle = all_gather_async(
                self.wo_b_tp_weight_scale,
                self.tp_group,
                output=AscendDSACPImpl.wo_b_full_weight_scale_pool,
            )
            if wo_b_weight_scale_handle is not None:
                handles.append(wo_b_weight_scale_handle)
        return handles

    def _switch_o_proj_to_full_weight(
        self,
        handles: list[torch.distributed.Work],
    ) -> None:
        for handle in handles:
            handle.wait()
        assert AscendDSACPImpl.wo_a_full_pool is not None
        self.wo_a.weight.set_(AscendDSACPImpl.wo_a_full_pool)
        if self._wo_a_dynamic_quant:
            assert AscendDSACPImpl.wo_a_full_weight_scale_pool is not None
            self.wo_a.weight_scale.set_(AscendDSACPImpl.wo_a_full_weight_scale_pool)
        assert AscendDSACPImpl.wo_b_full_pool is not None
        self.wo_b.weight.set_(AscendDSACPImpl.wo_b_full_pool)
        if self._wo_b_dynamic_quant:
            assert AscendDSACPImpl.wo_b_full_weight_scale_pool is not None
            self.wo_b.weight_scale.set_(AscendDSACPImpl.wo_b_full_weight_scale_pool)

    def _switch_o_proj_to_tp_weight(self) -> None:
        self.wo_a.weight.set_(self.wo_a_tp_weight)
        if self._wo_a_dynamic_quant:
            self.wo_a.weight_scale.set_(self.wo_a_tp_weight_scale)
        self.wo_b.weight.set_(self.wo_b_tp_weight)
        if self._wo_b_dynamic_quant:
            self.wo_b.weight_scale.set_(self.wo_b_tp_weight_scale)

    def _apply_wo_b(
        self,
        o_proj_input: torch.Tensor,
        full_weight: bool,
        skip_tp_reduce: bool = False,
    ) -> torch.Tensor:
        if not full_weight and not skip_tp_reduce:
            return self.wo_b(o_proj_input)
        return self.wo_b.quant_method.apply(self.wo_b, o_proj_input, bias=None)

    def _reduce_scatter_dsa_cp_o_proj_output(
        self,
        o_proj_output: torch.Tensor,
        local_cache_plan: DSACPLocalCachePlan,
        exchange_num_tokens: int,
    ) -> torch.Tensor:
        if self.tp_size == 1:
            return o_proj_output[: local_cache_plan.local_num_tokens]
        expected_num_tokens = self.tp_size * exchange_num_tokens
        if o_proj_output.shape[0] != expected_num_tokens:
            raise RuntimeError(
                "DSA CP local-cache o_proj reduce_scatter got unexpected token count: "
                f"expected={expected_num_tokens}, got={o_proj_output.shape[0]}."
            )
        reduced = torch.empty(
            (exchange_num_tokens, o_proj_output.shape[-1]),
            dtype=o_proj_output.dtype,
            device=o_proj_output.device,
        )
        dist.reduce_scatter_tensor(reduced, o_proj_output.contiguous(), group=self.tp_group.device_group)
        return reduced[: local_cache_plan.local_num_tokens]

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor],
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        assert attn_metadata[0].req_metadata is not None
        first_cp_metadata = attn_metadata[0].req_metadata.cp_metadata
        use_local_cache_cp = first_cp_metadata.local_cache_plan is not None
        full_gather_wo_a_enabled = (
            self.tp_size > 1
            and self.enable_dsa_cp_with_o_proj_tp
            and not use_local_cache_cp
            and attn_metadata[0].attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )
        local_attn_output, o_proj_full_handles = self._forward(
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv,
            full_gather_wo_a_enabled,
        )
        o_proj_input = self._restore_tp_head_layout(
            local_attn_output,
            layer_name,
            attn_metadata[0],
            skip_all_to_all=full_gather_wo_a_enabled,
        )
        if isinstance(o_proj_input, tuple):
            o_proj_input, dsa_cp_exchange_num_tokens = o_proj_input
        else:
            dsa_cp_exchange_num_tokens = None
        num_tokens = o_proj_input.shape[0]
        local_cache_plan = first_cp_metadata.local_cache_plan
        use_dsa_cp_local_output = (
            local_cache_plan is not None
            and dsa_cp_exchange_num_tokens is not None
            and not full_gather_wo_a_enabled
        )

        # o
        if full_gather_wo_a_enabled:
            self._switch_o_proj_to_full_weight(o_proj_full_handles)
        o_proj_groups = self.n_group if full_gather_wo_a_enabled else self.n_local_groups
        try:
            if get_ascend_device_type() in {AscendDeviceType.A5}:
                o = o_proj_input.view(num_tokens, o_proj_groups, -1)
                o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
                o = torch_npu.npu_transpose_quant_batchmatmul(
                    o,
                    self.wo_a.weight,
                    dtype=torch.bfloat16,
                    bias=None,
                    group_sizes=(0, 0, 32),
                    x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                    x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                )
                o = o.reshape(num_tokens, -1)
                o = self._apply_wo_b(o, full_gather_wo_a_enabled, skip_tp_reduce=use_dsa_cp_local_output)
                if use_dsa_cp_local_output:
                    assert local_cache_plan is not None
                    assert dsa_cp_exchange_num_tokens is not None
                    o = self._reduce_scatter_dsa_cp_o_proj_output(
                        o, local_cache_plan, dsa_cp_exchange_num_tokens
                    )
                output[...] = o
            else:
                o_proj_input = o_proj_input.view(num_tokens, o_proj_groups, -1)
                if olora_tp_enable():
                    o_proj_input = self.wo_a(o_proj_input)
                else:
                    # wo_a = self.wo_a.weight.view(o_proj_groups, self.o_lora_rank, -1)
                    # o = torch.einsum("tgd,grd->tgr", o, wo_a)
                    o_proj_input = torch_npu.npu_transpose_batchmatmul(
                        o_proj_input,
                        self.wo_a.weight,
                        bias=None,
                        scale=None,
                        perm_x1=(1, 0, 2),
                        perm_x2=(0, 1, 2),
                        perm_y=(1, 0, 2),
                        batch_split_factor=1,
                    )
                o_proj_input = o_proj_input.reshape(num_tokens, -1)
                o = self._apply_wo_b(
                    o_proj_input, full_gather_wo_a_enabled, skip_tp_reduce=use_dsa_cp_local_output
                )
                if use_dsa_cp_local_output:
                    assert local_cache_plan is not None
                    assert dsa_cp_exchange_num_tokens is not None
                    o = self._reduce_scatter_dsa_cp_o_proj_output(
                        o, local_cache_plan, dsa_cp_exchange_num_tokens
                    )
                output[...] = o
        finally:
            if full_gather_wo_a_enabled:
                self._switch_o_proj_to_tp_weight()

        return output

    def _get_tp_global_rank(self, group_rank: int) -> int:
        if hasattr(self.tp_group, "ranks"):
            return self.tp_group.ranks[group_rank]
        return dist.get_global_rank(self.tp_group.device_group, group_rank)

    @staticmethod
    def _select_dsa_cp_hidden_ranges(
        hidden_states: torch.Tensor,
        ranges: list[tuple[int, int]],
    ) -> torch.Tensor:
        if not ranges:
            return hidden_states[:0]
        if len(ranges) == 1:
            start, end = ranges[0]
            return hidden_states[start:end]
        return torch.cat([hidden_states[start:end] for start, end in ranges], dim=0)

    def _gather_dsa_cp_hidden_halos(
        self,
        hidden_states_local: torch.Tensor,
        local_cache_plan: DSACPLocalCachePlan | None,
        num_actual_tokens: int,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]] | None:
        if local_cache_plan is None or self.tp_size <= 1:
            return None
        if self.tp_group.device_group is None:
            raise RuntimeError("DSA CP local-cache hidden halo gather requires a TP process group.")
        if local_cache_plan.cp_size != self.tp_size:
            raise RuntimeError(
                "DSA CP local-cache hidden halo rank count mismatch: "
                f"expected {self.tp_size}, got {local_cache_plan.cp_size}."
            )

        halo_size = local_cache_plan.unit_size
        num_reqs = max(0, len(local_cache_plan.query_start_loc) - 1)
        local_tail = hidden_states_local.new_zeros((num_reqs * halo_size, *hidden_states_local.shape[1:]))
        local_offset_by_range = {
            local_range: local_offset
            for local_range, local_offset in zip(local_cache_plan.local_valid_ranges, local_cache_plan.local_offsets)
        }
        for req_idx, valid_range in enumerate(local_cache_plan.rank_valid_ranges[local_cache_plan.cp_rank]):
            valid_start, valid_end = valid_range
            valid_start = min(valid_start, num_actual_tokens)
            valid_end = min(valid_end, num_actual_tokens)
            if valid_start >= valid_end:
                continue
            compact_offset = local_offset_by_range.get((valid_start, valid_end))
            if compact_offset is None:
                continue
            tail_start = max(valid_start, valid_end - halo_size)
            tail_len = valid_end - tail_start
            local_read_start = compact_offset + tail_start - valid_start
            local_read_end = local_read_start + tail_len
            local_write_start = req_idx * halo_size
            local_tail[local_write_start : local_write_start + tail_len].copy_(
                hidden_states_local[local_read_start:local_read_end]
            )

        gathered_tails = [torch.empty_like(local_tail) for _ in range(local_cache_plan.cp_size)]
        dist.all_gather(gathered_tails, local_tail.contiguous(), group=self.tp_group.device_group)

        halo_buffers: list[torch.Tensor] = []
        halo_ranges: list[tuple[int, int]] = []
        for source_rank, rank_ranges in enumerate(local_cache_plan.rank_valid_ranges):
            for req_idx, (source_start, source_end) in enumerate(rank_ranges):
                source_start = min(source_start, num_actual_tokens)
                source_end = min(source_end, num_actual_tokens)
                if source_start >= source_end:
                    continue
                halo_start = max(source_start, source_end - halo_size)
                tail_offset = req_idx * halo_size
                halo_buffers.append(gathered_tails[source_rank][tail_offset : tail_offset + source_end - halo_start])
                halo_ranges.append((halo_start, source_end))

        return halo_buffers, halo_ranges

    def _assemble_dsa_cp_hidden_input(
        self,
        hidden_input_plan: DSACPHiddenInputPlan,
        hidden_states_local: torch.Tensor,
        hidden_halos: tuple[list[torch.Tensor], list[tuple[int, int]]] | None,
        hidden_states_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_input_plan.num_input_tokens == 0:
            return hidden_states_local[:0]

        if not hidden_input_plan.halo_source_ranges:
            return self._select_dsa_cp_hidden_ranges(hidden_states_local, hidden_input_plan.local_read_ranges)

        if hidden_halos is None:
            if hidden_states_full is not None:
                return self._select_dsa_cp_hidden_ranges(hidden_states_full, hidden_input_plan.input_ranges)
            raise RuntimeError("DSA CP hidden input requires halo data, but no halo source is available.")

        halo_buffers, halo_ranges = hidden_halos
        assembled = hidden_states_local.new_empty((hidden_input_plan.num_input_tokens, *hidden_states_local.shape[1:]))
        for (read_start, read_end), (out_start, out_end) in zip(
            hidden_input_plan.local_read_ranges, hidden_input_plan.local_output_ranges
        ):
            assembled[out_start:out_end].copy_(hidden_states_local[read_start:read_end])

        for (source_start, source_end), (out_start, out_end) in zip(
            hidden_input_plan.halo_source_ranges, hidden_input_plan.halo_output_ranges
        ):
            copied = False
            for buffer, (halo_start, halo_end) in zip(halo_buffers, halo_ranges):
                if halo_start <= source_start and source_end <= halo_end:
                    read_start = source_start - halo_start
                    read_end = read_start + source_end - source_start
                    assembled[out_start:out_end].copy_(buffer[read_start:read_end])
                    copied = True
                    break
            if not copied:
                if hidden_states_full is not None:
                    assembled[out_start:out_end].copy_(hidden_states_full[source_start:source_end])
                else:
                    raise RuntimeError(
                        "DSA CP hidden halo range is not covered by gathered rank tails: "
                        f"range=({source_start}, {source_end})."
                    )

        return assembled

    def _get_dsa_cp_request_tail_key(
        self,
        req_metadata_or_slot_plan: AscendDSAReqMetadata | DSACPCompressorSlotPlan,
        req_idx: int,
    ) -> str:
        request_ids = getattr(req_metadata_or_slot_plan, "request_ids", None)
        if request_ids is not None and req_idx < len(request_ids):
            return request_ids[req_idx]
        return f"idx:{req_idx}"

    def _assemble_dsa_cp_compressor_hidden_input(
        self,
        layer_name: str,
        slot_plan: DSACPCompressorSlotPlan,
        hidden_input_plan: DSACPHiddenInputPlan,
        hidden_states_local: torch.Tensor,
        hidden_halos: tuple[list[torch.Tensor], list[tuple[int, int]]] | None,
        hidden_states_full: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_hidden = self._assemble_dsa_cp_hidden_input(
            hidden_input_plan, hidden_states_local, hidden_halos, hidden_states_full
        )
        if not slot_plan.has_prefix_hidden:
            return current_hidden

        tail_cache = self._dsa_cp_hidden_tail_cache.get(layer_name)
        if tail_cache is None:
            raise RuntimeError(
                f"DSA CP local-cache layer {layer_name} requires previous hidden tail cache, but cache is empty."
            )

        prefix_lengths = slot_plan.prefix_lengths.tolist()
        request_indices = slot_plan.request_indices_cpu.tolist()
        current_start_positions = slot_plan.current_start_positions.tolist()
        pieces = []
        current_offset = 0
        for (input_start, input_end), prefix_len, req_idx, current_start_pos in zip(
            slot_plan.input_ranges, prefix_lengths, request_indices, current_start_positions
        ):
            current_len = input_end - input_start
            if prefix_len > 0:
                req_key = self._get_dsa_cp_request_tail_key(slot_plan, req_idx)
                if req_key not in tail_cache:
                    raise RuntimeError(
                        "DSA CP local-cache missing previous hidden tail for chunk prefill: "
                        f"layer={layer_name}, req_key={req_key}, prefix_len={prefix_len}."
                    )
                tail_end_pos, tail_hidden = tail_cache[req_key]
                if tail_end_pos != current_start_pos or tail_hidden.shape[0] < prefix_len:
                    raise RuntimeError(
                        "DSA CP local-cache previous hidden tail does not match current chunk: "
                        f"layer={layer_name}, req_key={req_key}, tail_end={tail_end_pos}, "
                        f"current_start={current_start_pos}, tail_len={tail_hidden.shape[0]}, prefix_len={prefix_len}."
                    )
                pieces.append(tail_hidden[-prefix_len:].to(device=current_hidden.device, dtype=current_hidden.dtype))
            pieces.append(current_hidden[current_offset : current_offset + current_len])
            current_offset += current_len

        if current_offset != current_hidden.shape[0]:
            raise RuntimeError(
                "DSA CP local-cache compressor hidden assembly consumed an unexpected number of current tokens: "
                f"consumed={current_offset}, available={current_hidden.shape[0]}."
            )
        if not pieces:
            return current_hidden[:0]
        return torch.cat(pieces, dim=0)

    def _save_dsa_cp_hidden_tail_cache(
        self,
        layer_name: str,
        hidden_states_local: torch.Tensor,
        hidden_halos: tuple[list[torch.Tensor], list[tuple[int, int]]] | None,
        req_metadata: AscendDSAReqMetadata,
        local_cache_plan: DSACPLocalCachePlan | None,
        num_actual_tokens: int,
    ) -> None:
        if local_cache_plan is None:
            self._dsa_cp_hidden_tail_cache.pop(layer_name, None)
            return
        input_positions_cpu = req_metadata.input_positions_cpu
        query_start_loc_cpu = req_metadata.query_start_loc_cpu
        if input_positions_cpu is None or query_start_loc_cpu is None:
            return
        query_start_list = query_start_loc_cpu.tolist()
        input_positions_list = input_positions_cpu.tolist()
        num_reqs = len(query_start_list) - 1
        tail_ranges: list[tuple[int, int]] = []
        tail_req_indices: list[int] = []
        tail_end_positions: list[int] = []
        for req_idx in range(num_reqs):
            req_start = min(query_start_list[req_idx], num_actual_tokens)
            req_end = min(query_start_list[req_idx + 1], num_actual_tokens)
            if req_start >= req_end:
                continue
            tail_start = max(req_start, req_end - local_cache_plan.unit_size)
            tail_ranges.append((tail_start, req_end))
            tail_req_indices.append(req_idx)
            tail_end_positions.append(input_positions_list[req_end - 1] + 1)

        cache: dict[str, tuple[int, torch.Tensor]] = {}
        if tail_ranges:
            tail_plan = build_dsa_cp_hidden_input_plan(tail_ranges, local_cache_plan, num_actual_tokens)
            tail_hidden = self._assemble_dsa_cp_hidden_input(tail_plan, hidden_states_local, hidden_halos)
            offset = 0
            for (tail_start, tail_end), req_idx, tail_end_pos in zip(tail_ranges, tail_req_indices, tail_end_positions):
                tail_len = tail_end - tail_start
                req_key = self._get_dsa_cp_request_tail_key(req_metadata, req_idx)
                cache[req_key] = (tail_end_pos, tail_hidden[offset : offset + tail_len].detach().clone())
                offset += tail_len
        self._dsa_cp_hidden_tail_cache[layer_name] = cache

    def _broadcast_dsa_cp_state_blocks(
        self,
        state_cache: torch.Tensor | None,
        state_broadcast_plan: DSACPStateBroadcastPlan | None,
    ) -> None:
        if state_cache is None or state_broadcast_plan is None or self.tp_size <= 1:
            return
        if self.tp_group.device_group is None or state_broadcast_plan.state_block_ids.numel() == 0:
            return

        source_ranks = state_broadcast_plan.source_ranks.tolist()
        state_block_ids = state_broadcast_plan.state_block_ids.tolist()
        state_valid_mask = state_broadcast_plan.state_valid_mask.tolist()
        for source_rank, state_block_id, state_valid in zip(source_ranks, state_block_ids, state_valid_mask):
            if not state_valid or source_rank < 0 or state_block_id < 0:
                continue

            state_block = state_cache[int(state_block_id)]
            if self.tp_rank == source_rank:
                buffer = state_block.clone()
            else:
                buffer = torch.empty_like(state_block)
            dist.broadcast(
                buffer,
                src=self._get_tp_global_rank(source_rank),
                group=self.tp_group.device_group,
            )
            state_block.copy_(buffer)

    def _all_gather_dsa_cp_cache_updates(
        self,
        local_update: torch.Tensor | None,
        slot_plan: DSACPCompressorSlotPlan | DSACPSWAWindowPlan | None,
        device: torch.device,
        update_dtype: torch.dtype,
        trailing_shape: tuple[int, ...],
        apply_update,
    ) -> None:
        if slot_plan is None or self.tp_size <= 1:
            return
        all_rank_update_counts = getattr(
            slot_plan,
            "all_rank_valid_output_counts",
            getattr(slot_plan, "all_rank_valid_token_counts", ()),
        )
        all_rank_slot_mappings = getattr(slot_plan, "all_rank_slot_mappings", ())
        if self.tp_group.device_group is None or not all_rank_update_counts:
            return
        if len(all_rank_update_counts) != self.tp_size:
            raise RuntimeError(
                "DSA CP local cache update plan rank count mismatch: "
                f"expected {self.tp_size}, got {len(all_rank_update_counts)}."
            )
        if len(all_rank_slot_mappings) != self.tp_size:
            raise RuntimeError(
                "DSA CP local cache slot mapping rank count mismatch: "
                f"expected {self.tp_size}, got {len(all_rank_slot_mappings)}."
            )

        max_update_count = max(all_rank_update_counts)
        if max_update_count <= 0:
            return

        local_count = all_rank_update_counts[self.tp_rank]
        if local_update is not None and local_update.shape[0] != local_count:
            raise RuntimeError(
                "DSA CP local cache update count mismatch: "
                f"expected {local_count}, got kv={local_update.shape[0]}."
            )
        if local_update is None and local_count != 0:
            raise RuntimeError(
                "DSA CP local cache update is missing for rank with valid outputs: "
                f"rank={self.tp_rank}, expected={local_count}."
            )

        if local_update is None:
            local_update = torch.zeros((0, *trailing_shape), dtype=update_dtype, device=device)
        else:
            local_trailing_shape = tuple(local_update.shape[1:])
            if local_trailing_shape != trailing_shape:
                raise RuntimeError(
                    "DSA CP local cache update trailing shape mismatch: "
                    f"expected={trailing_shape}, got={local_trailing_shape}."
                )
            if local_update.dtype != update_dtype:
                raise RuntimeError(
                    "DSA CP local cache update dtype mismatch: "
                    f"expected={update_dtype}, got={local_update.dtype}."
                )
            local_update = local_update.contiguous()

        padded_update = torch.zeros((max_update_count, *trailing_shape), dtype=update_dtype, device=device)
        if local_update.shape[0] > 0:
            padded_update[: local_update.shape[0]].copy_(local_update)

        gathered_updates = [torch.empty_like(padded_update) for _ in range(self.tp_size)]
        dist.all_gather(gathered_updates, padded_update, group=self.tp_group.device_group)

        for source_rank, update_count in enumerate(all_rank_update_counts):
            if update_count <= 0 or source_rank == self.tp_rank:
                continue
            source_update = gathered_updates[source_rank][:update_count]
            source_slot_mapping = all_rank_slot_mappings[source_rank].to(device=device)
            if source_slot_mapping.shape[0] != update_count:
                raise RuntimeError(
                    "DSA CP local cache slot mapping count mismatch: "
                    f"rank={source_rank}, expected={update_count}, got={source_slot_mapping.shape[0]}."
                )
            apply_update(source_update, source_slot_mapping)

    def _all_gather_dsa_cp_swa_cache_updates(
        self,
        swa_kv_cache: torch.Tensor | None,
        local_update: torch.Tensor | None,
        window_plan: DSACPSWAWindowPlan | None,
    ) -> None:
        if swa_kv_cache is None:
            return

        # SWA KV trailing shape is a model constant: (1, nope + rope).
        update_dtype = local_update.dtype if local_update is not None else self.kv_norm.weight.dtype

        def apply_update(update: torch.Tensor, slot_mapping: torch.Tensor) -> None:
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, update, slot_mapping)

        self._all_gather_dsa_cp_cache_updates(
            local_update,
            window_plan,
            swa_kv_cache.device,
            update_dtype,
            (1, self.nope_head_dim + self.rope_head_dim),
            apply_update,
        )

    def _all_gather_dsa_cp_compressed_cache_updates(
        self,
        compress_kv_cache: torch.Tensor | None,
        local_update: torch.Tensor | None,
        slot_plan: DSACPCompressorSlotPlan | None,
    ) -> None:
        if compress_kv_cache is None:
            return

        trailing_shape = (1, self.compressor_head_dim)
        update_dtype = self.compressor_norm.weight.dtype
        if local_update is not None:
            self._cached_compressor_kv_trailing = local_update.shape[1:]
            trailing_shape = self._cached_compressor_kv_trailing
            update_dtype = local_update.dtype
        elif self._cached_compressor_kv_trailing is not None:
            trailing_shape = self._cached_compressor_kv_trailing

        def apply_update(update: torch.Tensor, slot_mapping: torch.Tensor) -> None:
            DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, update, slot_mapping)

        self._all_gather_dsa_cp_cache_updates(
            local_update,
            slot_plan,
            compress_kv_cache.device,
            update_dtype,
            trailing_shape,
            apply_update,
        )

    def _all_gather_dsa_cp_indexer_cache_updates(
        self,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_full_cache: torch.Tensor | None,
        local_update: torch.Tensor | None,
        slot_plan: DSACPCompressorSlotPlan | None,
    ) -> None:
        trailing_shape = (1, self.indexcom_head_dim)
        update_dtype = self.indexcom_norm.weight.dtype
        if local_update is not None:
            self._cached_indexer_kv_trailing = local_update.shape[1:]
            trailing_shape = self._cached_indexer_kv_trailing
            update_dtype = local_update.dtype
        elif self._cached_indexer_kv_trailing is not None:
            trailing_shape = self._cached_indexer_kv_trailing

        def apply_update(update: torch.Tensor, slot_mapping: torch.Tensor) -> None:
            _, update_scale = DeviceOperator.indexer_quant_scatter_part1(
                update, indexer_k_cache, indexer_full_cache, slot_mapping
            )
            if update_scale is not None:
                DeviceOperator.dsa_indexer_scatter_scale_part3(update_scale, indexer_scale_cache, slot_mapping)

        self._all_gather_dsa_cp_cache_updates(
            local_update,
            slot_plan,
            indexer_k_cache.device,
            update_dtype,
            trailing_shape,
            apply_update,
        )

    def _forward(
        self,
        layer_name,
        hidden_states_local: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        full_gather_wo_a_enabled: bool = False,
    ):
        """Run full-sequence KV cache updates and local-token attention."""
        (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache, self.compress_ratio
        )
        if self.compress_ratio == 4:
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata
        elif self.compress_ratio == 128:
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
        else:
            (swa_metadata,) = attn_metadata
        common_attn_metadata = attn_metadata[0]
        hidden_states: torch.Tensor | None = None

        assert common_attn_metadata.req_metadata is not None
        assert swa_metadata.req_metadata is not None
        req_metadata = common_attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        cos = req_metadata.cos[layer_name]
        sin = req_metadata.sin[layer_name]
        local_cos = cp_metadata.local_cos[layer_name]
        local_sin = cp_metadata.local_sin[layer_name]
        actual_seq_lengths_query = req_metadata.query_start_loc
        local_seq_lengths_query = cp_metadata.local_query_start_loc
        local_seq_lengths_key = cp_metadata.local_seq_lens
        has_prefill = _has_prefill(common_attn_metadata.attn_state)
        use_local_cache_prefill = has_prefill and cp_metadata.local_cache_plan is not None
        hidden_halos = None
        if use_local_cache_prefill:
            hidden_halos = self._gather_dsa_cp_hidden_halos(
                hidden_states_local,
                cp_metadata.local_cache_plan,
                common_attn_metadata.num_actual_tokens,
            )
            hidden_states_cache = hidden_states_local
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)
            hidden_states_cache = hidden_states[: common_attn_metadata.num_actual_tokens]

        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and isinstance(
            self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod
        ):
            q_a = self.wq_a(hidden_states_local)
            qr_local, qr_pertoken_scale_local = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            if getattr(self.wq_b, "_chunk_size", 0):
                bias = self.wq_b.bias
                chunk_size = self.wq_b._chunk_size
                bias_1 = bias[:chunk_size] if bias is not None else None
                bias_2 = bias[chunk_size:] if bias is not None else None
                q = torch.cat(
                    [
                        torch_npu.npu_quant_matmul(
                            qr_local,
                            self.wq_b.weight_1,
                            self.wq_b.weight_1_scale,
                            pertoken_scale=qr_pertoken_scale_local,
                            bias=bias_1,
                            output_dtype=hidden_states_local.dtype,
                        ),
                        torch_npu.npu_quant_matmul(
                            qr_local,
                            self.wq_b.weight_2,
                            self.wq_b.weight_2_scale,
                            pertoken_scale=qr_pertoken_scale_local,
                            bias=bias_2,
                            output_dtype=hidden_states_local.dtype,
                        ),
                    ],
                    dim=-1,
                )
            else:
                q = torch_npu.npu_quant_matmul(
                    qr_local,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    pertoken_scale=qr_pertoken_scale_local,
                    bias=self.wq_b.bias,
                    output_dtype=hidden_states_local.dtype,
                )
        else:
            qr_local = self.q_norm(self.wq_a(hidden_states_local))
            q = self.wq_b(qr_local)
            qr_pertoken_scale_local = None

        q = q.unflatten(-1, (self.num_heads, self.head_dim))

        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            local_cos,
            local_sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        o_proj_full_handles = self._maybe_all_gather_o_proj_full_weight(full_gather_wo_a_enabled)

        swa_req_metadata = swa_metadata.req_metadata
        swa_cp_metadata = swa_req_metadata.cp_metadata
        swa_hidden_states_cache = hidden_states_cache
        swa_cos = cos
        swa_sin = sin
        swa_slot_mapping = swa_req_metadata.slot_mapping
        swa_window_plan = None
        if use_local_cache_prefill:
            assert swa_cp_metadata.local_cache_plan is not None
            swa_window_plan = swa_cp_metadata.swa_window_plan
            swa_hidden_states_cache = hidden_states_local
            if swa_cp_metadata.local_cache_plan.local_valid_ranges:
                swa_cos, swa_sin = concatenate_rope_slices(
                    cos,
                    swa_cp_metadata.local_cache_plan.local_valid_ranges,
                )
            else:
                swa_cos = cos[:0]
                swa_sin = sin[:0]
            assert swa_cp_metadata.swa_slot_mapping is not None
            swa_slot_mapping = swa_cp_metadata.swa_slot_mapping

        swa_kv = None
        if swa_hidden_states_cache.numel() > 0:
            swa_kv = self.wkv(swa_hidden_states_cache)
            swa_kv = self.kv_norm(swa_kv)
            assert self.rope_head_dim is not None
            swa_kv = swa_kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                swa_kv.unsqueeze(1),
                swa_cos[: swa_kv.shape[0]],
                swa_sin[: swa_kv.shape[0]],
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, swa_kv, swa_slot_mapping)

        if swa_window_plan is not None:
            self._all_gather_dsa_cp_swa_cache_updates(swa_kv_cache, swa_kv, swa_window_plan)

        compress_topk_idxs = None
        if self.compress_ratio > 1:
            assert compressor_attn_metadata.req_metadata is not None
            assert compressor_kv_state_metadata.req_metadata is not None
            compressor_cp_metadata = compressor_attn_metadata.req_metadata.cp_metadata
            compressor_slot_plan = (
                compressor_cp_metadata.compressor_slot_plan
                if has_prefill and compressor_cp_metadata.compressor_slot_plan is not None
                else None
            )
            if compressor_slot_plan is not None:
                compress_cos, compress_sin = get_cos_and_sin_dsa(
                    {f"c{self.compress_ratio}": compressor_slot_plan.compressed_positions},
                    use_cache=False,
                )
                compress_slot_mapping = compressor_slot_plan.slot_mapping
            else:
                compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(
                    compressor_attn_metadata.req_metadata,
                )

            if self.compress_ratio == 4:
                self._update_indexer_cache(
                    hidden_states_local=hidden_states_local,
                    hidden_halos=hidden_halos,
                    hidden_states_full=_select_indexer_hidden_states_full(
                        hidden_states,
                        hidden_states_cache,
                        use_local_cache_prefill,
                    ),
                    layer_name=layer_name,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                )
                compress_topk_idxs = self._indexer_select_topk(
                    x=hidden_states_local,
                    qr=qr_local,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=local_cos,
                    sin=local_sin,
                    actual_seq_lengths_query=local_seq_lengths_query,
                    actual_seq_lengths_key=local_seq_lengths_key,
                    qr_pertoken_scale=qr_pertoken_scale_local,
                )

            coff = 2 if self.compressor_overlap else 1
            compressor_hidden_states = hidden_states_cache
            compressor_cu_seqlens = actual_seq_lengths_query
            compressor_start_pos = req_metadata.start_pos
            compressor_state_block_table = compressor_kv_state_metadata.req_metadata.block_table
            compressor_slot_mapping = compress_slot_mapping
            compressor_sin = compress_sin
            compressor_cos = compress_cos
            compressor_state_broadcast_plan = compressor_kv_state_metadata.req_metadata.cp_metadata.state_broadcast_plan
            run_compressor = True
            if compressor_slot_plan is not None:
                run_compressor = compressor_slot_plan.input_indices.numel() > 0
                if run_compressor:
                    request_indices = compressor_slot_plan.request_indices.to(device=req_metadata.start_pos.device)
                    compressor_hidden_input_plan = compressor_cp_metadata.compressor_hidden_input_plan
                    if compressor_hidden_input_plan is not None:
                        compressor_hidden_states = self._assemble_dsa_cp_compressor_hidden_input(
                            layer_name,
                            compressor_slot_plan,
                            compressor_hidden_input_plan,
                            hidden_states_local,
                            hidden_halos,
                            hidden_states,
                        )
                    else:
                        if hidden_states is None:
                            raise RuntimeError("DSA CP compressor local-cache path requires a hidden input plan.")
                        input_indices = compressor_slot_plan.input_indices.to(device=hidden_states.device)
                        compressor_hidden_states = hidden_states[input_indices]
                    compressor_cu_seqlens = compressor_slot_plan.input_query_start_loc.to(
                        device=actual_seq_lengths_query.device
                    )
                    compressor_start_pos = req_metadata.start_pos.index_select(0, request_indices)
                    compressor_start_pos = compressor_start_pos + compressor_slot_plan.start_pos_offsets.to(
                        device=compressor_start_pos.device, dtype=compressor_start_pos.dtype
                    )
                    compressor_state_block_table = compressor_state_block_table.index_select(0, request_indices)

            if run_compressor:
                compressed_kv = torch.ops._C_ascend.compressor(
                    compressor_hidden_states,
                    self.compressor_wkv.weight,
                    self.compressor_wgate.weight,
                    state_cache.squeeze(-2),
                    self.compressor_ape,
                    self.compressor_norm.weight,
                    compressor_sin.view(-1, compressor_sin.shape[-1]),
                    compressor_cos.view(-1, compressor_cos.shape[-1]),
                    state_block_table=compressor_state_block_table,
                    cu_seqlens=compressor_cu_seqlens,
                    seqused=None,
                    start_pos=compressor_start_pos,
                    rope_head_dim=self.rope_head_dim,
                    cmp_ratio=self.compress_ratio,
                    coff=coff,
                    norm_eps=self.compressor_norm_eps,
                    rotary_mode=2,
                    cache_mode=1,
                )
            else:
                compressed_kv = hidden_states_cache[:0]

            if compressed_kv.numel() == 0:
                compressed_kv = None
            elif compressor_slot_plan is not None:
                valid_output_mask = compressor_slot_plan.valid_output_mask.to(device=compressed_kv.device)
                compressed_kv = compressed_kv[valid_output_mask]
                compressor_slot_mapping = compressor_slot_plan.slot_mapping[valid_output_mask]
                if compressed_kv.numel() == 0:
                    compressed_kv = None

            if compressed_kv is not None:
                DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compressor_slot_mapping)

            if compressor_slot_plan is not None:
                self._all_gather_dsa_cp_compressed_cache_updates(
                    compress_kv_cache, compressed_kv, compressor_slot_plan
                )

            self._broadcast_dsa_cp_state_blocks(state_cache, compressor_state_broadcast_plan)

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                extra_attn_kwargs, cu_seqlens_ori_kv=local_seq_lengths_query
            )

        common_attn_kwargs = dict(
            cu_seqlens_q=local_seq_lengths_query,
            seqused_kv=local_seq_lengths_key,
            sinks=self.attn_sink,
            softmax_scale=self.softmax_scale,
            cmp_ratio=max(self.compress_ratio, 1),
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            **extra_attn_kwargs,
        )

        if self.compress_ratio <= 1:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                metadata=swa_metadata.req_metadata.sas_metadata,
                **common_attn_kwargs,
            )[0]
        elif self.compress_ratio == 4:
            assert compressor_attn_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                metadata=req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        else:
            assert compressor_attn_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                metadata=compressor_attn_metadata.req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        if use_local_cache_prefill:
            self._save_dsa_cp_hidden_tail_cache(
                layer_name=layer_name,
                hidden_states_local=hidden_states_local,
                hidden_halos=hidden_halos,
                req_metadata=req_metadata,
                local_cache_plan=cp_metadata.local_cache_plan,
                num_actual_tokens=common_attn_metadata.num_actual_tokens,
            )

        return attn_output, o_proj_full_handles

    def _restore_tp_head_layout(
        self,
        local_attn_output: torch.Tensor,
        layer_name: str,
        attn_metadata: M,
        skip_all_to_all: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, int]:
        assert attn_metadata.req_metadata is not None
        req_metadata = attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        num_tokens = local_attn_output.shape[0]
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            local_attn_output.unsqueeze(1),
            cp_metadata.local_cos[layer_name],
            -cp_metadata.local_sin[layer_name],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        if self.tp_size == 1 or skip_all_to_all:
            return local_attn_output

        local_cache_plan = cp_metadata.local_cache_plan
        if local_cache_plan is None:
            send = (
                local_attn_output.view(num_tokens, self.tp_size, self.n_local_heads, self.head_dim)
                .permute(1, 0, 2, 3)
                .contiguous()
                .view(-1, self.n_local_heads, self.head_dim)
            )
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.tp_group.device_group)
            return recv

        all_rank_num_tokens = list(get_dsa_cp_all_rank_token_counts(local_cache_plan))
        expected_num_tokens = all_rank_num_tokens[self.tp_rank]
        if num_tokens != expected_num_tokens:
            raise RuntimeError(
                "DSA CP local-cache restore got unexpected local token count: "
                f"rank={self.tp_rank}, expected={expected_num_tokens}, got={num_tokens}."
            )
        exchange_num_tokens = max(all_rank_num_tokens, default=0)
        if exchange_num_tokens < num_tokens:
            raise RuntimeError(
                "DSA CP local-cache restore got invalid exchange token count: "
                f"exchange={exchange_num_tokens}, local={num_tokens}."
            )
        if exchange_num_tokens > num_tokens:
            local_attn_output = F.pad(local_attn_output, (0, 0, 0, 0, 0, exchange_num_tokens - num_tokens))
        send = (
            local_attn_output.view(exchange_num_tokens, self.tp_size, self.n_local_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, self.n_local_heads, self.head_dim)
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.tp_group.device_group)
        return recv, exchange_num_tokens

    def _update_indexer_cache(
        self,
        hidden_states_local: torch.Tensor,
        hidden_halos: tuple[list[torch.Tensor], list[tuple[int, int]]] | None,
        hidden_states_full: torch.Tensor | None,
        layer_name: str,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: list[M],
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
    ) -> None:
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata
        coff = 2 if self.compressor_overlap else 1
        assert indexer_kv_scale_metadata is not None
        assert indexer_kv_state_metadata is not None
        assert indexer_kv_scale_metadata.req_metadata is not None
        assert indexer_kv_state_metadata.req_metadata is not None
        assert self.indexer is not None
        indexer_state_req_metadata = indexer_kv_state_metadata.req_metadata
        indexer_scale_req_metadata = indexer_kv_scale_metadata.req_metadata
        indexer_slot_plan = indexer_scale_req_metadata.cp_metadata.compressor_slot_plan
        if indexer_slot_plan is None:
            indexer_compressed_cos, indexer_compressed_sin, indexer_slot_mapping = self._compute_compressor_metadata(
                indexer_scale_req_metadata,
            )
        else:
            indexer_compressed_cos = compressed_cos
            indexer_compressed_sin = compressed_sin
            indexer_slot_mapping = indexer_slot_plan.slot_mapping

        indexer_state_block_table = indexer_state_req_metadata.block_table
        indexer_cu_seqlens = actual_seq_lengths_query
        indexer_start_pos = indexer_scale_req_metadata.start_pos
        if hidden_states_full is None:
            indexer_x = hidden_states_local
        else:
            indexer_x = hidden_states_full
        state_broadcast_plan = indexer_state_req_metadata.cp_metadata.state_broadcast_plan
        run_indexer_compressor = True

        if indexer_slot_plan is not None:
            run_indexer_compressor = indexer_slot_plan.input_indices.numel() > 0
            if run_indexer_compressor:
                request_indices = indexer_slot_plan.request_indices.to(device=indexer_start_pos.device)
                indexer_hidden_input_plan = indexer_scale_req_metadata.cp_metadata.compressor_hidden_input_plan
                if indexer_hidden_input_plan is not None:
                    indexer_x = self._assemble_dsa_cp_compressor_hidden_input(
                        layer_name,
                        indexer_slot_plan,
                        indexer_hidden_input_plan,
                        hidden_states_local,
                        hidden_halos,
                        hidden_states_full,
                    )
                else:
                    if hidden_states_full is None:
                        raise RuntimeError("DSA CP indexer local-cache path requires a hidden input plan.")
                    input_indices = indexer_slot_plan.input_indices.to(device=hidden_states_full.device)
                    indexer_x = hidden_states_full[input_indices]
                indexer_cu_seqlens = indexer_slot_plan.input_query_start_loc.to(device=actual_seq_lengths_query.device)
                indexer_start_pos = indexer_start_pos.index_select(0, request_indices)
                indexer_start_pos = indexer_start_pos + indexer_slot_plan.start_pos_offsets.to(
                    device=indexer_start_pos.device, dtype=indexer_start_pos.dtype
                )
                indexer_state_block_table = indexer_state_block_table.index_select(0, request_indices)

        kv = None
        if run_indexer_compressor:
            kv = torch.ops._C_ascend.compressor(
                indexer_x,
                self.indexcom_wkv.weight,
                self.indexcom_wgate.weight,
                indexer_state_cache.squeeze(-2),
                self.indexcom_ape,
                self.indexcom_norm.weight,
                indexer_compressed_sin.view(-1, indexer_compressed_sin.shape[-1]),
                indexer_compressed_cos.view(-1, indexer_compressed_cos.shape[-1]),
                state_block_table=indexer_state_block_table,
                cu_seqlens=indexer_cu_seqlens,
                seqused=None,
                start_pos=indexer_start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

        if kv is not None and kv.numel() > 0:
            if indexer_slot_plan is not None:
                valid_output_mask = indexer_slot_plan.valid_output_mask.to(device=kv.device)
                kv = kv[valid_output_mask]
                indexer_slot_mapping = indexer_slot_plan.slot_mapping[valid_output_mask]
                if kv.numel() == 0:
                    kv = None
            if kv is not None:
                if self.indexer.compressor.rotate:
                    kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

                _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
                    kv,
                    indexer_k_cache,
                    indexer_full_cache,
                    indexer_slot_mapping,
                )
                if kv_scale is not None:
                    DeviceOperator.dsa_indexer_scatter_scale_part3(
                        kv_scale,
                        indexer_scale_cache,
                        indexer_slot_mapping,
                    )

        if indexer_slot_plan is not None:
            self._all_gather_dsa_cp_indexer_cache_updates(
                indexer_k_cache,
                indexer_scale_cache,
                indexer_full_cache,
                kv,
                indexer_slot_plan,
            )

        self._broadcast_dsa_cp_state_blocks(indexer_state_cache, state_broadcast_plan)

    def _indexer_select_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: list[M],
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (_, indexer_k_cache, indexer_scale_cache, _) = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        (_, _, _, indexer_kv_scale_metadata, _) = attn_metadata
        assert indexer_kv_scale_metadata is not None

        if (
            (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod))
            and isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and qr_pertoken_scale is not None
            and get_ascend_device_type() not in {AscendDeviceType.A5}
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )
        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        q, q_scale = DeviceOperator.indexer_quantize_query(q)

        assert indexer_kv_scale_metadata.req_metadata is not None
        qli_metadata = indexer_kv_scale_metadata.req_metadata.qli_metadata
        block_table = indexer_kv_scale_metadata.req_metadata.block_table
        topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
            key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
            actual_seq_lengths_query=actual_seq_lengths_query[1:],
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_idxs

    def dsa_warmup_with_multistream(self, hidden_states: torch.Tensor):
        pass
