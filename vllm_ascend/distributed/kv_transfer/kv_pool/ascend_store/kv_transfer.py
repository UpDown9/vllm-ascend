from __future__ import annotations

import ctypes
import math
import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerTransferTask,
    ReqMeta,
    SharedBlockData,
    get_block_hashes,
)
# isort: on


DSA_CP_PREFIX_CACHE_UNIT_SIZE = 128


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


def _circular_shift_array(value: np.ndarray, offset: int) -> np.ndarray:
    length = len(value)
    if length == 0:
        return value
    offset %= length
    if offset == 0:
        return value
    return np.concatenate((value[offset:], value[:offset]))


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        my_key_index: int,
        num_ranks_per_layer: int,
        page_size_bytes: int,
        num_layers: int,
    ) -> None:
        self.my_key_index = my_key_index
        self.num_ranks_per_layer = num_ranks_per_layer
        self.page_size_bytes = page_size_bytes
        self.num_layers = num_layers
        self._block_len_np = np.asarray(token_database.group_block_len[0], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[0],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(0, token_database.group_block_len[0])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        # group_block_len[0] / kv_caches_base_addr[0] are laid out flat as
        # [layer0_caches..., layer1_caches..., ...]; the per-layer stride is the
        # total length divided by the number of layers (mirrors
        # ChunkedTokenDatabase caches_per_layer computation).
        self._caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
        self._block_ids_buf: np.ndarray | None = None
        self._block_gvas_buf: np.ndarray | None = None

    def _ensure_buf(self, capacity: int) -> tuple[np.ndarray, np.ndarray]:
        if self._block_ids_buf is None or len(self._block_ids_buf) < capacity:
            self._block_ids_buf = np.empty(capacity, dtype=np.int64)
            self._block_gvas_buf = np.empty(capacity, dtype=np.int64)
        assert self._block_ids_buf is not None and self._block_gvas_buf is not None
        return self._block_ids_buf[:capacity], self._block_gvas_buf[:capacity]

    @staticmethod
    def _dedupe_transfer_blocks(
        block_ids_arr: np.ndarray,
        block_gvas_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_ids_arr.size <= 1:
            return block_ids_arr, block_gvas_arr

        block_transfer_array = np.column_stack((block_ids_arr, block_gvas_arr))
        _, unique_indices = np.unique(
            block_transfer_array,
            axis=0,
            return_index=True,
        )
        if unique_indices.size == block_ids_arr.size:
            return block_ids_arr, block_gvas_arr

        return (
            block_ids_arr[unique_indices],
            block_gvas_arr[unique_indices],
        )

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        caches_per_layer = self._caches_per_layer
        # group_* arrays are laid out flat as [layer0_caches..., layer1_caches...];
        # slice the per-layer window for ``layer_id``. Using the full length as the
        # stride (the old behaviour) overshoots for layer_id >= 1 and yields empty
        # slices -> broadcast errors.
        base_offset = layer_id * caches_per_layer
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset : base_offset + caches_per_layer]
        layer_block_len = self._block_len_np[base_offset : base_offset + caches_per_layer]
        layer_block_stride = self._block_stride_np[base_offset : base_offset + caches_per_layer]
        # Per-cache inner offsets within one layer's page: [0, len0, len0+len1, ...].
        layer_inner_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
        )
        rank_layer_offset = (layer_id * self.num_ranks_per_layer + self.my_key_index) * self.page_size_bytes

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
        )

    @staticmethod
    def _require_request_arrays(
        block_range: LayerBlockRange,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        if request.block_ids_np is None or request.block_gvas_np is None:
            raise RuntimeError("ReqMeta numpy block metadata is not initialized")
        return request.block_ids_np, request.block_gvas_np

    def build_shared(self, task: LayerTransferTask) -> SharedBlockData | None:
        """Pre-compute shared block data that is identical across all layers."""
        if not task.block_ranges:
            return None

        total = 0
        for block_range in task.block_ranges:
            total += block_range.end_block - block_range.start_block
            if block_range.partial_block_index is not None:
                total += 1

        block_ids_arr, block_gvas_arr = self._ensure_buf(total)
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        offset = 0

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range)

            num_blocks = block_range.end_block - block_range.start_block
            if num_blocks > 0:
                gva_start = block_range.start_block - request.gva_block_offset
                gva_end = block_range.end_block - request.gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {request.gva_block_offset}"
                    )
                end = offset + num_blocks
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                assert request.last_block_gva is not None
                block_ids_arr[offset] = block_ids_np[block_range.partial_block_index]
                block_gvas_arr[offset] = request.last_block_gva
                offset += 1

        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(block_ids_arr[:offset], block_gvas_arr[:offset])

        return SharedBlockData(
            block_ids_arr=block_ids_arr,
            block_gvas_arr=block_gvas_arr,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
        )

    def build_addrs(
        self,
        shared: SharedBlockData,
        layer_id: int,
    ) -> LayerBatchReqMeta:
        """Compute per-layer addresses from pre-computed shared block data."""
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            shared.block_ids_arr, shared.block_gvas_arr, layer_id
        )

        return LayerBatchReqMeta(
            req_ids=shared.req_ids,
            layer_id=layer_id,
            is_last_chunks=shared.is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
        )

    def build(self, task: LayerTransferTask) -> LayerBatchReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task)
        if shared is None:
            return None
        return self.build_addrs(shared, task.layer_id)


class KVTransferThread(threading.Thread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. This indicates queue shutdown or invalid request.")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

    def _store_mask(self, req_meta: ReqMeta) -> tuple[list[bool], ...] | None:
        store_mask = getattr(self.token_database, "store_mask", None)
        if store_mask is None:
            return None
        try:
            return store_mask(req_meta.token_len_chunk, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_meta.req_id, exc)
            return None

    def _load_mask(self, req_meta: ReqMeta, token_len: int) -> tuple[list[bool], ...] | None:
        load_mask = getattr(self.token_database, "load_mask", None)
        if load_mask is None:
            return None
        return load_mask(req_meta.block_hashes, token_len)

    def _mask_allows_chunk(
        self,
        masks: tuple[list[bool], ...] | None,
        group_id: int,
        start: int,
    ) -> bool:
        mask_allows_chunk = getattr(self.token_database, "mask_allows_chunk", None)
        if mask_allows_chunk is None:
            return True
        return mask_allows_chunk(masks, group_id, start)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        group_uses_align_state: list[bool],
        enable_kv_event: bool = False,
        group_uses_swa: list[bool] | None = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        validate_put_cache_blocks: Callable[
            [int, list[int], str, list[str]], None
        ]
        | None = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state or []
        self.group_uses_swa = group_uses_swa or []
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.validate_put_cache_blocks = validate_put_cache_blocks
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}
        self.put_error_lock = threading.Lock()
        self.put_error: RuntimeError | None = None

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def mark_completed_events(self, event_id: int | None) -> None:
        if event_id is not None:
            with self.completed_events_lock:
                self.completed_events[event_id] = 1

    def get_completed_events(self):
        if not self.completed_events:
            return None
        with self.completed_events_lock:
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def raise_put_error(self) -> None:
        with self.put_error_lock:
            error = self.put_error
            self.put_error = None
        if error is not None:
            raise error

    @staticmethod
    def _get_dsa_cp_owner_range(
        token_len: int, cp_size: int, cp_rank: int, unit_size: int = DSA_CP_PREFIX_CACHE_UNIT_SIZE
    ) -> tuple[int, int]:
        num_units = math.ceil(token_len / unit_size)
        base_units = num_units // cp_size
        remainder_units = num_units % cp_size
        local_units = base_units + int(cp_rank < remainder_units)
        local_start_units = cp_rank * base_units + min(cp_rank, remainder_units)
        local_start = local_start_units * unit_size
        return min(local_start, token_len), min(local_start + local_units * unit_size, token_len)

    def _filter_dsa_cp_swa_owner_blocks(
        self,
        starts: list[int],
        ends: list[int],
        keys: list[str],
        block_hashes: list,
        group_id: int,
        token_len: int,
        owner_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[list[int], list[int], list[str], list]:
        if group_id >= len(self.group_uses_swa) or not self.group_uses_swa[group_id]:
            return starts, ends, keys, block_hashes

        if owner_ranges is None:
            if self.cp_size <= 1:
                return starts, ends, keys, block_hashes
            owner_ranges = [self._get_dsa_cp_owner_range(token_len, self.cp_size, self.cp_rank)]
        owner_ranges = [(start, end) for start, end in owner_ranges if start < end]
        filtered_starts: list[int] = []
        filtered_ends: list[int] = []
        filtered_keys: list[str] = []
        filtered_hashes: list = []
        skipped_partial = 0
        for start, end, key, block_hash in zip(starts, ends, keys, block_hashes):
            if any(owner_start <= end - 1 < owner_end for owner_start, owner_end in owner_ranges):
                # A cache-transfer block is owned by the rank containing its
                # last token. This also handles a transfer-granularity block
                # whose boundary window crosses CP ranks, while keeping
                # ownership unique and avoiding duplicate external puts.
                filtered_starts.append(start)
                filtered_ends.append(end)
                filtered_keys.append(key)
                filtered_hashes.append(block_hash)
            elif any(start < owner_end and end > owner_start for owner_start, owner_end in owner_ranges):
                skipped_partial += 1
        if skipped_partial:
            logger.warning(
                "Skip %d partial DSA CP SWA prefix-cache blocks for group %d; "
                "expected put chunks to be owner-range aligned.",
                skipped_partial,
                group_id,
            )
        return filtered_starts, filtered_ends, filtered_keys, filtered_hashes

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            with self.put_error_lock:
                if self.put_error is not None:
                    return
            if req_id not in self.stored_requests:
                logger.debug(
                    "TEST KV pool put skipped req=%s reason=request_not_tracked token_len=%d",
                    req_id,
                    token_len,
                )
                return

            store_masks = self._store_mask(req_meta)
            for group_id in req_meta.kv_cache_group_ids or [0]:
                starts = []
                ends = []
                keys = []
                block_hashes = []
                key_block_ids = []
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    group_block_size,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )

                for start, end, key, block_id in self._process_tokens_with_block_ids(
                    token_len,
                    req_meta.block_hashes,
                    block_ids,
                    kv_cache_group_id=group_id,
                    skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                ):
                    if not self._mask_allows_chunk(store_masks, group_id, start):
                        continue
                    starts.append(start)
                    ends.append(end)
                    keys.append(key.to_string())
                    block_hashes.append(group_block_hashes[start // group_block_size])
                    key_block_ids.append(block_id)

                block_ids_by_start = dict(zip(starts, key_block_ids))
                starts, ends, keys, block_hashes = self._filter_dsa_cp_swa_owner_blocks(
                    starts,
                    ends,
                    keys,
                    block_hashes,
                    group_id,
                    token_len,
                    getattr(req_meta, "dsa_cp_swa_owner_ranges", None),
                )
                key_block_ids = [block_ids_by_start[start] for start in starts]

                use_explicit_swa_owner = (
                    group_id < len(self.group_uses_swa)
                    and self.group_uses_swa[group_id]
                    and req_meta.dsa_cp_swa_owner_ranges is not None
                )
                if (
                    not self.dcp_size > 1
                    and not req_meta.disable_tp_key_sharding
                    and not use_explicit_swa_owner
                    and not self.group_uses_align_state[group_id]
                ):
                    starts = starts[self.tp_rank % self.put_step :: self.put_step]
                    ends = ends[self.tp_rank % self.put_step :: self.put_step]
                    keys = keys[self.tp_rank % self.put_step :: self.put_step]
                    block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]
                    key_block_ids = key_block_ids[self.tp_rank % self.put_step :: self.put_step]

                if not keys:
                    logger.debug(
                        "TEST KV pool put skipped req=%s group=%d reason=no_keys token_len=%d "
                        "block_hashes=%d block_ids=%d put_step=%d tp_rank=%d",
                        req_id,
                        group_id,
                        token_len,
                        len(req_meta.block_hashes),
                        len(block_ids),
                        self.put_step,
                        self.tp_rank,
                    )
                    continue

                exists_states = self.lookup(keys)
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                if not missing_indices:
                    logger.debug(
                        "TEST KV pool put skipped req=%s group=%d reason=all_keys_exist token_len=%d "
                        "keys=%d sample_keys=%s",
                        req_id,
                        group_id,
                        token_len,
                        len(keys),
                        keys[:3],
                    )
                    continue

                starts = [starts[index] for index in missing_indices]
                ends = [ends[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]
                block_hashes = [block_hashes[index] for index in missing_indices]
                key_block_ids = [key_block_ids[index] for index in missing_indices]

                # logger.info(
                #     "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                #     len(keys),
                #     token_len // group_block_size,
                #     len(missing_indices),
                #     req_id,
                #     group_id,
                # )
                logger.debug(
                    "TEST KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                    req_id,
                    group_id,
                    token_len,
                    len(keys),
                    keys[:3],
                )

                addrs = []
                sizes = []
                stored_events: list[BlockStored] = []
                prev_key = None
                new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
                for index, start in enumerate(starts):
                    addr, size, _ = self._prepare_value(
                        start,
                        ends[index],
                        block_ids,
                        kv_cache_group_id=group_id,
                        block_id=key_block_ids[index],
                    )
                    addrs.append(addr)
                    sizes.append(size)

                    # Create KV event
                    if self.enable_kv_event:
                        token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                        block_size = (
                            req_meta.original_block_size[group_id]
                            if isinstance(req_meta.original_block_size, list)
                            else req_meta.original_block_size
                        )
                        if block_size is not None:
                            stored_event = BlockStored(
                                block_hashes=[new_block_hashes[index]],
                                parent_block_hash=prev_key,
                                token_ids=token_ids,
                                block_size=block_size,
                                lora_id=None,
                                medium="cpu",
                                lora_name=None,
                            )
                            stored_events.append(stored_event)
                            prev_key = new_block_hashes[index]
                            logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                if self.kv_role == "kv_consumer":
                    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                        keys,
                        addrs,
                        sizes,
                        kv_cache_group_id=group_id,
                    )

                if current_event is not None:
                    current_event.synchronize()
                if self.validate_put_cache_blocks is not None:
                    try:
                        self.validate_put_cache_blocks(
                            group_id,
                            key_block_ids,
                            req_id,
                            keys,
                        )
                    except RuntimeError as error:
                        with self.put_error_lock:
                            self.put_error = error
                        raise
                # logger.info(
                #     "[KV-STORE-TRACE] operation=put request_id=%s "
                #     "group_id=%d tp_rank=%d keys=%s",
                #     req_id,
                #     group_id,
                #     self.tp_rank,
                #     keys,
                # )
                self.m_store.put(keys, addrs, sizes)

                # TODO Query specific replica info to update the event
                if self.enable_kv_event and stored_events is not None:
                    self.update_kv_event(stored_events)
        finally:
            # always free blocks
            self.mark_completed_events(req_meta.event_id)
            self.dec_stored_request(req_id)
            self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        block_id_list: list[int] = []
        group_ids = req_meta.kv_cache_group_ids or [0]
        load_masks = self._load_mask(req_meta, token_len)
        for group_id in group_ids:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            for start, end, key, block_id in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                if not self._mask_allows_chunk(load_masks, group_id, start):
                    continue
                addr, size, block_id = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=block_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        block_id_list_c = (
            block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
        )
        logger.debug(
            "TEST KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
            key_list_c[:3],
        )
        # logger.info(
        #     "[KV-STORE-TRACE] operation=load request_id=%s mode=async "
        #     "groups=%s tp_rank=%d keys=%s",
        #     req_id,
        #     group_ids,
        #     self.tp_rank,
        #     key_list_c,
        # )
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        # logger.info(
        #     "[KV-STORE-TRACE] operation=load_result request_id=%s "
        #     "mode=async tp_rank=%d key_status=%s",
        #     req_id,
        #     self.tp_rank,
        #     list(zip(key_list_c, ret, strict=False))
        #     if ret is not None
        #     else None,
        # )
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                ret,
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        elif ret is None:
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                [1] * len(block_id_list_c),
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        logger.debug(
            "TEST KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
        )
        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
        self.stored_requests: dict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()
        self.layerwise_event_lock = threading.Lock()

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
        with self.layerwise_event_lock:
            self.layerwise_event_starts.pop(req_id, None)

    def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]) -> None:
        if self.enable_kv_event:
            with self.layerwise_event_lock:
                self.layerwise_event_starts[req_meta.req_id].update(starts)

    def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
        if not self.enable_kv_event or req_meta.layer_id != self.final_layer_id:
            return []
        block_size = (
            req_meta.original_block_size[req_meta.kv_cache_group_id]
            if isinstance(req_meta.original_block_size, list)
            else req_meta.original_block_size
        )
        if block_size is None:
            return []
        stored_events: list[BlockStored] = []
        group_block_size = self._get_block_size(req_meta.kv_cache_group_id)
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes]
        with self.layerwise_event_lock:
            starts_set = self.layerwise_event_starts.pop(req_meta.req_id, set())
        for start in sorted(starts_set):
            block_idx = start // group_block_size
            if block_idx >= len(new_block_hashes):
                continue
            block_hash = new_block_hashes[block_idx]
            parent_block_hash = new_block_hashes[block_idx - 1] if block_idx > 0 else None
            end = min(start + group_block_size, len(req_meta.token_ids or []))
            token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
            stored_event = BlockStored(
                block_hashes=[block_hash],
                parent_block_hash=parent_block_hash,
                token_ids=token_ids,
                block_size=block_size,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            )
            stored_events.append(stored_event)
            logger.debug("Added layerwise kv cache event '%s' to kv cache events queue", stored_event)
        return stored_events

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        starts = req_meta.starts
        ends = req_meta.ends
        keys = req_meta.keys
        layer_id = req_meta.layer_id
        current_event = req_meta.current_event
        total_block = len(keys)
        is_last_chunk = req_meta.is_last_chunk
        log_layerwise_put = logger.debug
        with self.done_task_lock:
            if req_meta.req_id not in self.stored_requests:
                log_layerwise_put(
                    "TEST KV pool layerwise put skipped req=%s layer=%d reason=request_not_tracked total_blocks=%d",
                    req_meta.req_id,
                    layer_id,
                    total_block,
                )
                self.request_queue.task_done()
                return
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            log_layerwise_put(
                "TEST KV pool layerwise put skipped req=%s layer=%d reason=no_keys_after_shard "
                "total_blocks=%d put_step=%d tp_rank=%d is_last_chunk=%s",
                req_meta.req_id,
                layer_id,
                total_block,
                self.put_step,
                self.tp_rank,
                is_last_chunk,
            )
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk and layer_id == self.final_layer_id:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        exists_states = self.lookup(key_list)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            log_layerwise_put(
                "TEST KV pool layerwise put skipped req=%s layer=%d reason=all_keys_exist "
                "total_blocks=%d keys=%d is_last_chunk=%s sample_keys=%s",
                req_meta.req_id,
                layer_id,
                total_block,
                len(key_list),
                is_last_chunk,
                key_list[:3],
            )
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk and layer_id == self.final_layer_id:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
            return

        starts = [starts[index] for index in missing_indices]
        ends = [ends[index] for index in missing_indices]
        key_list = [key_list[index] for index in missing_indices]

        addr_list = []
        size_list = []
        for index, key in enumerate(key_list):
            addr, size, _ = self.token_database.prepare_value_layer(
                starts[index], ends[index], req_meta.block_ids_by_group[0], layer_id
            )
            addr_list.append(addr)
            size_list.append(size)

        log_layerwise_put(
            "TEST KV pool layerwise put request=%s layer=%d total_blocks=%d missing=%d "
            "keys=%d is_last_chunk=%s sample_keys=%s",
            req_meta.req_id,
            layer_id,
            total_block,
            len(missing_indices),
            len(key_list),
            is_last_chunk,
            key_list[:3],
        )
        if current_event is not None:
            current_event.synchronize()
        self.m_store.put(key_list, addr_list, size_list)
        self._record_layerwise_event_starts(req_meta, starts)
        stored_events = self._build_stored_events(req_meta)
        if stored_events:
            self.update_kv_event(stored_events)

        if layer_id == self.final_layer_id and is_last_chunk:
            with self.layerwise_event_lock:
                self.layerwise_event_starts.pop(req_meta.req_id, None)
            self.dec_stored_request(req_meta.req_id)
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        # if layer_id == self.final_layer_id:
        #     logger.info(
        #         "TEST Storing KV cache layerwise for %d out of %d blocks (missing_count=%d) for request %s, layer %d",
        #         len(key_list),
        #         total_block,
        #         len(missing_indices),
        #         req_meta.req_id,
        #         layer_id,
        #     )
        # else:
        #     logger.debug(
        #         "Storing KV cache layerwise for %d out of %d blocks (missing_count=%d) for request %s, layer %d",
        #         len(key_list),
        #         total_block,
        #         len(missing_indices),
        #         req_meta.req_id,
        #         layer_id,
        #     )


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        addr_list = []
        size_list = []
        key_list = []
        block_id_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size, block_id = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index], req_meta.block_ids_by_group[0], req_meta.layer_id
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
            block_id_list.append(block_id)

        offset = self.tp_rank % len(key_list)
        key_list_c = key_list[offset:] + key_list[:offset]
        addr_list_c = addr_list[offset:] + addr_list[:offset]
        size_list_c = size_list[offset:] + size_list[:offset]
        block_id_list_c = block_id_list[offset:] + block_id_list[:offset]
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)

        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                ret,
            )
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        elif ret is None:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(block_id_list_c)

        self.request_queue.task_done()
        self.get_event.set()


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
