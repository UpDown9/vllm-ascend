# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from collections.abc import Iterable

import vllm.v1.core.block_pool
import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    _approximate_gcd,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.private_swa_pool import (
    PrivateSWAConfig,
    compute_private_swa_prefill_workspace_blocks,
    is_private_swa_kv_cache_spec,
)
from vllm_ascend import envs
from vllm_ascend.utils import vllm_version_is


def _queue_block_summary(block: KVCacheBlock) -> str:
    prev_id = block.prev_free_block.block_id if block.prev_free_block is not None else None
    next_id = block.next_free_block.block_id if block.next_free_block is not None else None
    return (
        f"block_id={block.block_id} ref_cnt={block.ref_cnt} "
        f"is_null={block.is_null} prev_free_block={prev_id} next_free_block={next_id}"
    )


def _swa_block_diag(kind: str, block: KVCacheBlock, where: str) -> None:
    msg = f"SWA_BLOCK_DIAG {kind} where={where} {_queue_block_summary(block)}"
    logger.warning(msg)


def _dedupe_free_blocks(blocks: Iterable[KVCacheBlock], where: str) -> list[KVCacheBlock]:
    deduped_blocks: list[KVCacheBlock] = []
    seen_block_ids: set[int] = set()
    for block in blocks:
        if not block.is_null and block.block_id in seen_block_ids:
            _swa_block_diag("duplicate_free_batch", block, where)
            continue
        if not block.is_null:
            seen_block_ids.add(block.block_id)
        deduped_blocks.append(block)
    return deduped_blocks


def _filter_queue_insert_blocks(blocks: list[KVCacheBlock], where: str) -> list[KVCacheBlock]:
    filtered_blocks: list[KVCacheBlock] = []
    seen_block_ids: set[int] = set()
    for block in blocks:
        if block.is_null:
            _swa_block_diag("null_free_queue_insert", block, where)
            continue
        if block.block_id in seen_block_ids:
            _swa_block_diag("duplicate_free_queue_insert", block, where)
            continue
        if block.ref_cnt != 0:
            _swa_block_diag("nonzero_ref_cnt_free_queue_insert", block, where)
            continue
        if block.prev_free_block is not None or block.next_free_block is not None:
            _swa_block_diag("linked_free_queue_insert", block, where)
            continue
        seen_block_ids.add(block.block_id)
        filtered_blocks.append(block)
    return filtered_blocks


_orig_block_pool_free_blocks = BlockPool.free_blocks


def _ascend_free_blocks(
    self: BlockPool,
    ordered_blocks: Iterable[KVCacheBlock],
    prepend: bool = False,
) -> None:
    filtered_blocks: list[KVCacheBlock] = []
    for block in _dedupe_free_blocks(ordered_blocks, "BlockPool.free_blocks"):
        if not block.is_null and block.ref_cnt <= 0:
            _swa_block_diag("ref_cnt_underflow_free_blocks", block, "BlockPool.free_blocks")
            continue
        filtered_blocks.append(block)
    _orig_block_pool_free_blocks(self, filtered_blocks, prepend)


_orig_free_queue_prepend_n = FreeKVCacheBlockQueue.prepend_n
_orig_free_queue_append_n = FreeKVCacheBlockQueue.append_n


def _ascend_free_queue_prepend_n(self: FreeKVCacheBlockQueue, blocks: list[KVCacheBlock]) -> None:
    _orig_free_queue_prepend_n(self, _filter_queue_insert_blocks(blocks, "FreeKVCacheBlockQueue.prepend_n"))


def _ascend_free_queue_append_n(self: FreeKVCacheBlockQueue, blocks: list[KVCacheBlock]) -> None:
    _orig_free_queue_append_n(self, _filter_queue_insert_blocks(blocks, "FreeKVCacheBlockQueue.append_n"))


_ORIGINAL_GROUP_AND_UNIFY = vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs
_ORIGINAL_UNIFORM_GROUPS = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups
_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_ORIGINAL_PACKED_CONFIG = getattr(
    vllm.v1.core.kv_cache_utils, "_get_kv_cache_config_packed",
    getattr(vllm.v1.core.kv_cache_utils, "_get_kv_cache_config_deepseek_v4", None),
)

_ORIGINAL_GET_KV_CACHE_CONFIGS = vllm.v1.core.kv_cache_utils.get_kv_cache_configs


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL:
        return _ORIGINAL_GROUP_AND_UNIFY(kv_cache_spec)

    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda r: (r != 4, r)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL:
        return _ORIGINAL_UNIFORM_GROUPS(grouped_specs)
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Plan shared DeepSeek-V4 caches plus a physically separate SWA pool.

    Private SWA layers are deliberately omitted from packed/shared tensors.
    Each receives exactly ``2 * max_num_seqs`` fixed ring allocations and is
    therefore independent of the shared BlockPool's block count.
    """
    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL:
        if _ORIGINAL_PACKED_CONFIG is None:
            raise RuntimeError("vLLM packed KV cache planner is unavailable")
        return _ORIGINAL_PACKED_CONFIG(vllm_config, kv_cache_groups, available_memory)

    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())

    private_layers: list[tuple[str, KVCacheSpec]] = []
    mtp_layer_names: list[str] = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        bucket: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            spec = specs[name]
            if (isinstance(spec, SlidingWindowMLASpec)
                    and getattr(spec, "model_version", None) == "deepseek_v4"):
                private_layers.append((name, spec))
            elif "mtp" in name:
                mtp_layer_names.append(name)
                mtp_page_size = spec.page_size_bytes
            else:
                bucket[spec.page_size_bytes].append(name)
        bucketed.append(bucket)

    if not private_layers:
        raise RuntimeError("DeepSeek-V4 cache layout has no private SWA layer")
    first_private = private_layers[0][1]
    if any((spec.block_size, spec.sliding_window, spec.page_size_bytes)
           != (first_private.block_size, first_private.sliding_window,
               first_private.page_size_bytes)
           for _, spec in private_layers):
        raise RuntimeError("all DeepSeek-V4 private SWA layers must share one layout")

    draft_tokens = (getattr(vllm_config.speculative_config,
                            "num_speculative_tokens", 0)
                    if vllm_config.speculative_config is not None else 0)
    private_config = PrivateSWAConfig(
        block_size=first_private.block_size,
        window_size=first_private.sliding_window,
        in_flight_tokens=1 + draft_tokens,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
    )
    private_bytes = (len(private_layers) * private_config.num_blocks
                     * first_private.page_size_bytes)
    # Prefill forwards are eager and serialized, so all DSV4 SWA layers share
    # one address-stable PA workspace. Reserve the same bound used at runtime:
    # per request, max(W-1 history, W rollback) plus page-alignment slack.
    prefill_workspace_blocks = compute_private_swa_prefill_workspace_blocks(
        max_num_batched_tokens=(
            vllm_config.scheduler_config.max_num_batched_tokens
        ),
        max_num_seqs=private_config.max_num_seqs,
        window_size=private_config.window_size,
        block_size=private_config.block_size,
    )
    prefill_workspace_bytes = (
        prefill_workspace_blocks * first_private.page_size_bytes)
    reserved_private_bytes = private_bytes + prefill_workspace_bytes
    if reserved_private_bytes >= available_memory:
        raise ValueError(
            "Insufficient KV memory for DeepSeek-V4 private SWA pool and "
            "Prefill workspace: "
            f"private={private_bytes}, workspace={prefill_workspace_bytes}, "
            f"available={available_memory}")

    # Only shared cache layers participate in the normal profiled budget.
    shared_page_sizes = sorted({ps for bucket in bucketed for ps in bucket})
    layer_tuple_page_bytes = sum(shared_page_sizes)
    shared_tuple_count = max(
        (len(layers) for bucket in bucketed for layers in bucket.values()),
        default=0,
    )
    total_shared_tuples = shared_tuple_count + len(mtp_layer_names)
    if layer_tuple_page_bytes <= 0 or total_shared_tuples <= 0:
        raise RuntimeError("DeepSeek-V4 cache layout has no shared cache layers")
    num_blocks = ((available_memory - reserved_private_bytes)
                  // (layer_tuple_page_bytes * total_shared_tuples))
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(shared_tuple_count):
        for page_size in shared_page_sizes:
            shared_by: list[str] = []
            for bucket in bucketed:
                layers = bucket.get(page_size)
                if layers is not None and tuple_idx < len(layers):
                    shared_by.append(layers[tuple_idx])
            if shared_by:
                kv_cache_tensors.append(KVCacheTensor(
                    size=page_size * num_blocks, shared_by=shared_by))
    for name in mtp_layer_names:
        kv_cache_tensors.append(KVCacheTensor(
            size=mtp_page_size * num_blocks, shared_by=[name]))
    # Append fixed-size private SWA tensors only after vLLM has aligned the
    # shared tensors across ranks. get_kv_cache_configs() assumes every tensor
    # scales with num_blocks, which is intentionally false for this ring pool.

    logger.info(
        "DeepSeek-V4 private SWA pool: layers=%d allocations=%d "
        "blocks_per_allocation=%d blocks=%d bytes=%d "
        "prefill_workspace_blocks=%d prefill_workspace_bytes=%d "
        "shared_blocks=%d",
        len(private_layers), private_config.num_allocations,
        private_config.blocks_per_allocation, private_config.num_blocks,
        private_bytes, prefill_workspace_blocks, prefill_workspace_bytes,
        num_blocks)
    return num_blocks, kv_cache_tensors


_PRIVATE_SWA_TENSOR_ATTR = "_vllm_ascend_private_swa"


def _private_swa_config_for(
    vllm_config: VllmConfig, spec: SlidingWindowMLASpec
) -> PrivateSWAConfig:
    draft_tokens = (
        getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
        if vllm_config.speculative_config is not None
        else 0
    )
    return PrivateSWAConfig(
        block_size=spec.block_size,
        window_size=spec.sliding_window,
        in_flight_tokens=1 + draft_tokens,
        max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
    )


def _iter_private_swa_layers(
    kv_cache_config: KVCacheConfig,
) -> list[tuple[str, SlidingWindowMLASpec]]:
    private_layers: list[tuple[str, SlidingWindowMLASpec]] = []
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        specs = getattr(group_spec, "kv_cache_specs", None)
        for layer_name in group.layer_names:
            layer_spec = specs[layer_name] if isinstance(specs, dict) else group_spec
            if (
                isinstance(layer_spec, SlidingWindowMLASpec)
                and is_private_swa_kv_cache_spec(layer_spec)
            ):
                private_layers.append((layer_name, layer_spec))
    return private_layers


def _append_private_swa_tensors(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Append fixed-size private tensors after shared-pool rank alignment."""
    private_layers = _iter_private_swa_layers(kv_cache_config)
    if not private_layers:
        return
    existing_layers = {
        layer_name
        for tensor in kv_cache_config.kv_cache_tensors
        for layer_name in tensor.shared_by
    }
    first_config = _private_swa_config_for(vllm_config, private_layers[0][1])
    for layer_name, layer_spec in private_layers:
        if layer_name in existing_layers:
            continue
        private_config = _private_swa_config_for(vllm_config, layer_spec)
        if private_config != first_config:
            raise RuntimeError("all DeepSeek-V4 private SWA layers must share one layout")
        tensor = KVCacheTensor(
            size=layer_spec.page_size_bytes * private_config.num_blocks,
            shared_by=[layer_name],
        )
        # The dynamic marker survives normal Python process transfer. Workers
        # also derive private membership from the layer spec as a safe fallback.
        setattr(tensor, _PRIVATE_SWA_TENSOR_ATTR, True)
        kv_cache_config.kv_cache_tensors.append(tensor)


def _ascend_get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Keep private-SWA tensor sizes out of shared rank-size normalization."""
    kv_cache_configs = _ORIGINAL_GET_KV_CACHE_CONFIGS(
        vllm_config, kv_cache_specs, available_memory
    )
    if envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL:
        for kv_cache_config in kv_cache_configs:
            _append_private_swa_tensors(vllm_config, kv_cache_config)
    return kv_cache_configs


BlockPool.free_blocks = _ascend_free_blocks
vllm.v1.core.block_pool.BlockPool.free_blocks = _ascend_free_blocks
FreeKVCacheBlockQueue.prepend_n = _ascend_free_queue_prepend_n
FreeKVCacheBlockQueue.append_n = _ascend_free_queue_append_n
vllm.v1.core.kv_cache_utils.FreeKVCacheBlockQueue.prepend_n = _ascend_free_queue_prepend_n
vllm.v1.core.kv_cache_utils.FreeKVCacheBlockQueue.append_n = _ascend_free_queue_append_n
vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
vllm.v1.core.kv_cache_utils.get_kv_cache_configs = _ascend_get_kv_cache_configs
# vllm v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to _get_kv_cache_config_packed and
# get_kv_cache_config_from_groups now calls _get_kv_cache_config_packed directly, bypassing
# the alias patch above. Patch the canonical name so Ascend's non-packed layout is used.
if vllm_version_is("0.23.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
else:
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.engine.core.get_kv_cache_configs = _ascend_get_kv_cache_configs
