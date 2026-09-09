# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import sys
from collections.abc import Mapping
from math import lcm

import vllm
import vllm.envs as envs_vllm
import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from vllm_ascend import envs
from vllm_ascend.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec

USE_MULTI_GROUPS_KV_CACHE = True

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _is_deepseek_v4_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    if getattr(kv_cache_spec, "model_version", None) == "deepseek_v4":
        return True

    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if nested_specs is None:
        return False

    if isinstance(nested_specs, Mapping):
        nested_specs = nested_specs.values()
    elif not isinstance(nested_specs, (list, tuple, set)):
        return False

    return any(_is_deepseek_v4_kv_cache_spec(spec) for spec in nested_specs)


def _is_deepseek_v4_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    return any(_is_deepseek_v4_kv_cache_spec(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups)


def _is_private_swa_group_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """Return whether a cache group consists exclusively of DSV4 SWA specs."""
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return getattr(kv_cache_spec, "model_version", None) == "deepseek_v4"
    nested = getattr(kv_cache_spec, "kv_cache_specs", None)
    if isinstance(nested, Mapping):
        nested = tuple(nested.values())
    elif isinstance(nested, (list, tuple, set)):
        nested = tuple(nested)
    else:
        return False
    return bool(nested) and all(_is_private_swa_group_spec(spec) for spec in nested)



class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
        scheduler_block_size: int | None = None,
    ):
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.retention_interval = getattr(envs_vllm, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
        validate_retention_interval = getattr(
            vllm_kv_cache_coordinator,
            "_validate_prefix_cache_retention_interval",
            None,
        )
        if self.retention_interval is not None and validate_retention_interval is not None:
            validate_retention_interval(
                self.retention_interval,
                self.scheduler_block_size,
                kv_cache_config,
            )
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        extra_mgr_kwargs: dict = {"scheduler_block_size": scheduler_block_size}
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                **extra_mgr_kwargs,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        self.private_swa_group_ids = (
            frozenset(
                i
                for i, manager in enumerate(self.single_type_managers)
                if _is_private_swa_group_spec(manager.kv_cache_spec)
            ) if envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL else frozenset()
        )
        self.private_swa_pool = None
        self.private_swa_config = None

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                self._get_effective_block_size(g.kv_cache_spec) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
            ), "block_size must be divisible by hash_block_size"
        self.verify_and_split_kv_cache_groups()

        # Align the WRITE-path mask granularity (reachable_block_mask) with the
        # READ-path hit granularity (find_longest_cache_hit) so SlidingWindowManager
        # only caches blocks that land on a boundary where future cache hits can
        # actually be matched.
        # TODO (Csrayz): Consider unified all single_type_managers to simplify logic.
        for mgr in self.single_type_managers:
            if isinstance(mgr, SlidingWindowManager):
                mgr.scheduler_block_size = self.lcm_block_size

        self.use_eagle = use_eagle

    def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        block_size = kv_cache_spec.block_size
        if isinstance(kv_cache_spec, MambaSpec) and self.enable_caching:
            return block_size
        if self.dcp_world_size * self.pcp_world_size > 1:
            block_size *= self.dcp_world_size * self.pcp_world_size
        if hasattr(kv_cache_spec, "compress_ratio"):
            compress_ratio = kv_cache_spec.compress_ratio or 1
            compress_ratio = compress_ratio if compress_ratio >= 1 else 1
            block_size *= compress_ratio
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

        # Propagate the eagle bit to every manager in an eagle-containing
        # attention group, mirroring upstream
        # HybridKVCacheCoordinator.verify_and_split_kv_cache_groups. Managers
        # default to ``use_eagle=False`` ("initialized lazily by the
        # coordinator", see SingleTypeKVCacheManager.__init__).
        #
        # Required for prefix-cache correctness on DeepSeek-V4 + MTP/EAGLE: the
        # SWA write path (``cache_blocks`` -> ``reachable_block_mask``) keys the
        # retained checkpoint tail on ``manager.use_eagle``, while the read path
        # (``find_longest_cache_hit``) applies ``drop_eagle_block`` to every gid
        # merged into the eagle attention group (and ``get_cached_block``
        # requires the block cached for *all* of them). If any such manager
        # keeps the default False, its retained tail ends one block short of the
        # eagle "peek" boundary the read looks at, the SWA group never hits, and
        # the min-over-groups hybrid hit collapses to 0%. Note the upstream
        # ``_annotate_eagle_groups_deepseek_v4`` flags only the single group
        # holding the MTP layer, so iterating ``eagle_group_ids`` alone would
        # miss its same-spec siblings.
        for idx in self.eagle_attn_group_indices:
            for gid in self.attention_groups[idx][1]:
                self.single_type_managers[gid].use_eagle = True

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: use 16k as the alignment tokens for model with compress ratio
        block_sizes = [self._get_effective_block_size(spec) for spec, _, _ in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def cache_blocks(self, request, num_computed_tokens: int) -> None:
        """Cache shared groups while keeping request-private SWA unhashed."""
        for group_id, manager in enumerate(self.single_type_managers):
            if group_id in self.private_swa_group_ids:
                continue
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )

    def get_num_blocks_to_allocate(
        self, request_id, num_tokens, new_computed_blocks, num_encoder_tokens,
        total_computed_tokens, num_tokens_main_model,
        apply_admission_cap=False,
    ) -> int:
        """Count only shared-pool blocks; private SWA is reserved atomically."""
        total = 0
        for group_id, manager in enumerate(self.single_type_managers):
            if group_id in self.private_swa_group_ids:
                continue
            if isinstance(manager, CrossAttentionManager):
                total += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [], 0,
                    num_encoder_tokens, apply_admission_cap=apply_admission_cap)
            else:
                total += manager.get_num_blocks_to_allocate(
                    request_id, num_tokens, new_computed_blocks[group_id],
                    total_computed_tokens, num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap)
        return total

    def allocate_new_computed_blocks(
        self, request_id, new_computed_blocks, num_local_computed_tokens,
        num_external_computed_tokens,
    ) -> None:
        """Attach prefix-hit and external blocks for shared cache groups.

        ``SingleTypeKVCacheManager`` owns the complete local-hit/external-KV
        allocation protocol. In particular, it performs the running-request
        fast path and handles sliding-window null blocks. Keep private SWA
        groups out of that protocol because their blocks are request-private.
        """
        for group_id, manager in enumerate(self.single_type_managers):
            if group_id in self.private_swa_group_ids:
                continue
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[group_id],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self, request_id, num_tokens, num_tokens_main_model,
        num_encoder_tokens=0,
    ):
        result = []
        for gid, manager in enumerate(self.single_type_managers):
            if gid in self.private_swa_group_ids:
                result.append([])
            else:
                result.append(manager.allocate_new_blocks(
                    request_id,
                    num_encoder_tokens if isinstance(manager, CrossAttentionManager)
                    else num_tokens,
                    num_tokens_main_model))
        return tuple(result)

    def free(self, request_id: str) -> None:
        for gid, manager in enumerate(self.single_type_managers):
            if gid not in self.private_swa_group_ids:
                manager.free(request_id)

    def remove_skipped_blocks(
        self, request_id: str, processed_computed_tokens: int,
    ) -> None:
        for gid, manager in enumerate(self.single_type_managers):
            if gid not in self.private_swa_group_ids:
                manager.remove_skipped_blocks(
                    request_id, processed_computed_tokens)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [
            0 if gid in self.private_swa_group_ids else
            manager.get_num_common_prefix_blocks(running_request_id)
            for gid, manager in enumerate(self.single_type_managers)
        ]

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            target_block_size = kv_cache_spec.block_size
            if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
                target_block_size *= self.dcp_world_size * self.pcp_world_size
            if target_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, target_block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                if any(group_id in self.private_swa_group_ids for group_id in group_ids):
                    for group_id in group_ids:
                        hit_blocks_by_group[group_id] = []
                    continue
                effective_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    num_blocks = curr_hit_length // effective_block_size
                    curr_hit_length = num_blocks * effective_block_size
                    continue

                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

                _max_length = curr_hit_length
                if use_eagle and not isinstance(spec, MambaSpec):
                    # Mamba finders do not drop the EAGLE lookahead block, so
                    # allowing a margin here could grow the hybrid hit length.
                    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
                eagle_kwarg = {"drop_eagle_block": use_eagle}
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    **eagle_kwarg,
                    alignment_tokens=self.lcm_block_size,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
                )
                _new_hit_length = len(hit_blocks[0]) * effective_block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                curr_hit_length = len(hit_blocks[0]) * effective_block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        # NOTE(zxr): for deepseek-v4, there is two fullattn groups, but
        # in this function, only the first fullattn group is truncate by
        # the belowing codes(c4), c128 layer does not truncate, which may
        # have prefix cache block hit.
        # Due to slidingwindow attn, deepseek-v4 decode node can't have
        # any prefix cache hit, because `hit_length` of SWA is 0.
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            target_block_size = kv_cache_spec.block_size
            if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
                target_block_size *= self.dcp_world_size * self.pcp_world_size
            if target_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, target_block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()
        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                if any(group_id in self.private_swa_group_ids for group_id in group_ids):
                    for group_id in group_ids:
                        hit_blocks_by_group[group_id] = []
                    continue
                # In PD disaggregation, Mamba running/temporal state is transferred
                # via the KV connector, but the D side has no local Mamba prefix
                # cache hit. If we let Mamba groups participate in the min-reduction,
                # their zero hit collapses the FullAttention hit length to 0 and
                # defeats prefix caching on the D side. Skip them instead.
                if isinstance(spec, MambaSpec):
                    if hit_blocks_by_group[group_ids[0]] is None:
                        for gid in group_ids:
                            hit_blocks_by_group[gid] = []
                    continue

                effective_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    num_blocks = curr_hit_length // effective_block_size
                    curr_hit_length = num_blocks * effective_block_size
                    continue

                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

                _max_length = curr_hit_length
                if use_eagle and not isinstance(spec, MambaSpec):
                    # Mamba finders do not drop the EAGLE lookahead block, so
                    # allowing a margin here could grow the hybrid hit length.
                    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
                eagle_kwarg = {"drop_eagle_block": use_eagle}
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    **eagle_kwarg,
                    alignment_tokens=self.lcm_block_size,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
                )
                _new_hit_length = len(hit_blocks[0]) * effective_block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                curr_hit_length = len(hit_blocks[0]) * effective_block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        # NOTE(zxr): for deepseek-v4, there is two fullattn groups, but
        # in this function, only the first fullattn group is truncate by
        # the belowing codes(c4), c128 layer does not truncate, which may
        # have prefix cache block hit.
        # Due to slidingwindow attn, deepseek-v4 decode node can't have
        # any prefix cache hit, because `hit_length` of SWA is 0.
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    scheduler_block_size: int | None = None,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        return AscendHybridKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
            max_num_batched_tokens=max_num_batched_tokens,
            scheduler_block_size=scheduler_block_size,
        )

    if len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        orig_kwargs = dict(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        orig_kwargs["scheduler_block_size"] = scheduler_block_size
        return _orig_get_kv_cache_coordinator(**orig_kwargs)

    return AscendHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_num_batched_tokens=max_num_batched_tokens,
        scheduler_block_size=scheduler_block_size,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
