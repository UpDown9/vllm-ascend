# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextvars import ContextVar
from functools import wraps
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

from vllm_ascend import envs
from vllm_ascend.core.private_swa_pool import (
    PrivateSWAAllocationPool,
    PRIVATE_SWA_TRANSFER_FAILED,
    PRIVATE_SWA_TRANSFER_NONE,
    PRIVATE_SWA_TRANSFER_READY,
    PRIVATE_SWA_TRANSFER_RECEIVING,
    PrivateSWAConfig,
    compute_prefix_rollback_start,
    detect_new_prefix_hit_length,
)

_prune_context: ContextVar[tuple[str, int] | None] = ContextVar(
    "ascend_swa_prune_context", default=None
)


def _find_private_swa_spec(spec: Any) -> SlidingWindowSpec | None:
    if isinstance(spec, SlidingWindowSpec) and getattr(spec, "model_version", None) == "deepseek_v4":
        return spec
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict):
        nested = nested.values()
    if isinstance(nested, (list, tuple, set)):
        for child in nested:
            found = _find_private_swa_spec(child)
            if found is not None:
                return found
    return None


def _handle_negative_in_flight(
    request_id: str,
    num_in_flight_tokens: int,
    where: str,
    num_scheduled_tokens: int | None = None,
) -> None:
    msg = (
        "SWA_BLOCK_DIAG negative_in_flight_tokens "
        f"where={where} request_id={request_id} "
        f"num_in_flight_tokens={num_in_flight_tokens}"
    )
    if num_scheduled_tokens is not None:
        msg += f" num_scheduled_tokens={num_scheduled_tokens}"
    logger.warning(msg)


def _safe_in_flight_tokens(request: Request, where: str) -> int:
    num_in_flight_tokens = getattr(request, "num_in_flight_tokens", 0)
    if num_in_flight_tokens < 0:
        _handle_negative_in_flight(
            request.request_id,
            num_in_flight_tokens,
            where,
        )
        request.num_in_flight_tokens = 0
        return 0
    return num_in_flight_tokens


def _max_in_flight_tokens(vllm_config: VllmConfig) -> int:
    return vllm_config.max_concurrent_batches * vllm_config.scheduler_config.max_num_batched_tokens


_original_request_init = Request.__init__


@wraps(_original_request_init)
def _patched_request_init(self: Request, *args: Any, **kwargs: Any) -> None:
    _original_request_init(self, *args, **kwargs)
    self.num_in_flight_tokens = 0
    self.private_swa_allocation = None
    self.private_swa_confirmed_length = 0
    self.private_swa_valid_length = 0
    self.private_swa_window_start = 0
    self.private_swa_pool_type = "private"
    self.private_swa_original_hit_length = None
    self.private_swa_effective_start = None
    self.private_swa_transfer_allocation = None
    self.private_swa_transfer_state = PRIVATE_SWA_TRANSFER_NONE
    self.private_swa_imported_window_start = None
    self.private_swa_imported_valid_length = None
    self.private_swa_transfer_shared_block_ids = frozenset()


def _record_private_swa_prefix_hit(
    request: Request,
    hit_length: int,
    private_pool: PrivateSWAAllocationPool,
) -> None:
    """Record one-shot SWA rollback boundaries for a cache hit."""
    if not envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL or hit_length <= 0:
        return
    previous_hit_length = getattr(request, "private_swa_original_hit_length", None)
    request.private_swa_original_hit_length = int(hit_length)
    request.private_swa_effective_start = compute_prefix_rollback_start(
        int(hit_length), private_pool.config.window_size
    )
    # A local hit can be observed both during get_computed_blocks and during
    # the post-schedule reconciliation. Log only the first observation to
    # avoid duplicate records for the same request.
    if previous_hit_length != int(hit_length):
        total_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
        rollback_start = int(request.private_swa_effective_start)
        logger.info(
            "PRIVATE_SWA_POOL prefix_hit request_id=%s total_prompt_tokens=%d "
            "hit_tokens=%d rollback_start=%d rollback_tokens=%d",
            request.request_id,
            total_prompt_tokens,
            int(hit_length),
            rollback_start,
            int(hit_length) - rollback_start,
        )


def _private_swa_import_covers(
    request: Request,
    confirmed_length: int,
    window_size: int,
) -> bool:
    imported_start = getattr(
        request, "private_swa_imported_window_start", None
    )
    imported_end = getattr(
        request, "private_swa_imported_valid_length", None
    )
    if imported_start is None or imported_end is None:
        return False
    required_start = max(0, confirmed_length - window_size + 1)
    return int(imported_start) <= required_start and int(imported_end) >= confirmed_length


_original_get_computed_blocks = KVCacheManager.get_computed_blocks


@wraps(_original_get_computed_blocks)
def _patched_get_computed_blocks(self: KVCacheManager, request: Request):
    """Record local prefix hits for every scheduler implementation."""
    was_uncomputed = int(request.num_computed_tokens) == 0
    result = _original_get_computed_blocks(self, request)
    if was_uncomputed and envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL:
        private_pool = getattr(self.coordinator, "private_swa_pool", None)
        if private_pool is not None:
            _, hit_length = result
            _record_private_swa_prefix_hit(request, int(hit_length), private_pool)
    elif was_uncomputed:
        # Keep prefix-cache observability when the private SWA feature is
        # disabled. No rollback state is created on this compatibility path.
        _, hit_length = result
        if int(hit_length) > 0:
            total_prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
            logger.info(
                "PREFIX_CACHE hit request_id=%s total_prompt_tokens=%d "
                "hit_tokens=%d private_swa_pool_enabled=False",
                request.request_id,
                total_prompt_tokens,
                int(hit_length),
            )
    return result


KVCacheManager.get_computed_blocks = _patched_get_computed_blocks


_original_update_after_schedule = Scheduler._update_after_schedule


@wraps(_original_update_after_schedule)
def _patched_update_after_schedule(
    self: Scheduler, scheduler_output: SchedulerOutput
) -> None:
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_swa_pool", None
    )
    computed_before_schedule: dict[str, int] = {}
    if private_pool is not None:
        # Capture the scheduler-visible position before upstream advances it.
        # A zero value identifies newly admitted requests whose prefix may be
        # resolved by an external connector rather than get_computed_blocks().
        computed_before_schedule = {
            request_id: int(request.num_computed_tokens)
            for request_id, request in self.requests.items()
        }
    _original_update_after_schedule(self, scheduler_output)
    private_metadata: dict[str, dict[str, Any]] = {}
    if private_pool is not None:
        # Reconcile the final scheduler-visible hit after local and external
        # matching. External async loads may emit no num_scheduled_tokens entry
        # (WAITING_FOR_REMOTE_KVS); retain H/R until the request runs next.
        for request_id, computed_before in computed_before_schedule.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            status = getattr(request, "status", None)
            status_name = getattr(status, "name", str(status))
            num_scheduled = int(
                scheduler_output.num_scheduled_tokens.get(request_id, 0)
            )
            confirmed_length = max(
                0, int(request.num_computed_tokens) - num_scheduled
            )
            transfer_state = getattr(
                request, "private_swa_transfer_state",
                PRIVATE_SWA_TRANSFER_NONE,
            )
            if transfer_state == PRIVATE_SWA_TRANSFER_RECEIVING:
                continue
            if transfer_state == PRIVATE_SWA_TRANSFER_FAILED:
                continue
            if transfer_state == PRIVATE_SWA_TRANSFER_READY:
                if not _private_swa_import_covers(
                    request, confirmed_length, private_pool.config.window_size
                ):
                    request.private_swa_transfer_state = (
                        PRIVATE_SWA_TRANSFER_FAILED
                    )
                    raise RuntimeError(
                        "imported private SWA does not cover the confirmed "
                        f"prefix for request {request_id}"
                    )
                # The D node must consume P-generated SWA directly. Clear any
                # provisional rollback recorded while the async load waited.
                request.private_swa_original_hit_length = None
                request.private_swa_effective_start = None
                continue
            confirmed_from_schedule = detect_new_prefix_hit_length(
                computed_before,
                int(request.num_computed_tokens),
                num_scheduled,
                waiting_for_remote_kvs=status_name == "WAITING_FOR_REMOTE_KVS",
            )
            if confirmed_from_schedule is not None:
                _record_private_swa_prefix_hit(
                    request, confirmed_from_schedule, private_pool
                )
    for request_id, num_scheduled_tokens in (
        scheduler_output.num_scheduled_tokens.items()
    ):
        request = self.requests.get(request_id)
        if request is None:
            # Upstream may remove an immediately finished request while applying
            # the schedule update. There is no metadata to publish in that case.
            continue
        request.num_in_flight_tokens += num_scheduled_tokens
        if private_pool is None:
            continue
        allocation = private_pool.get_allocation(request_id)
        if allocation is None:
            raise RuntimeError(f"request {request_id} has no private SWA allocation")
        # _update_after_schedule has already advanced num_computed_tokens to P.
        # Private metadata must describe the scheduler-visible prefix H.
        confirmed = max(
            0, int(request.num_computed_tokens) - int(num_scheduled_tokens)
        )
        original_hit = getattr(request, "private_swa_original_hit_length", None)
        effective_start = getattr(request, "private_swa_effective_start", None)
        valid = confirmed + int(num_scheduled_tokens)
        window = private_pool.config.window_size
        window_start = max(0, valid - window)
        request.private_swa_allocation = allocation
        request.private_swa_confirmed_length = confirmed
        request.private_swa_valid_length = valid
        request.private_swa_window_start = window_start
        private_metadata[request_id] = {
            "layout_version": 1,
            "pool_type": "private",
            "allocation_handle": allocation,
            "confirmed_length": confirmed,
            "shared_cache_length": confirmed,
            "swa_compute_start": (
                effective_start if effective_start is not None else confirmed
            ),
            "rollback_length": (
                confirmed - effective_start
                if effective_start is not None else 0
            ),
            # Compatibility aliases for the model metadata path.
            "original_prefix_hit_length": original_hit,
            "effective_compute_start": (
                effective_start if effective_start is not None else confirmed
            ),
            "valid_length": valid,
            "window_start": window_start,
            "physical_block_size": private_pool.config.block_size,
            "window_size": window,
            "in_flight_tokens": private_pool.config.in_flight_tokens,
            "blocks_per_allocation": private_pool.config.blocks_per_allocation,
            "num_allocations": private_pool.config.num_allocations,
            "private_num_blocks": private_pool.config.num_blocks,
        }
        # Prefix rollback metadata and imported-SWA readiness are one-shot.
        # After the first D execution, subsequent chunks use the locally
        # maintained private ring as ordinary history.
        if (
            getattr(request, "private_swa_transfer_state", None)
            == PRIVATE_SWA_TRANSFER_READY
        ):
            request.private_swa_transfer_state = PRIVATE_SWA_TRANSFER_NONE
        request.private_swa_original_hit_length = None
        request.private_swa_effective_start = None
    setattr(scheduler_output, "private_swa_metadata", private_metadata)


_original_update_from_output = Scheduler.update_from_output


_original_handle_invalid_blocks = Scheduler._handle_invalid_blocks


@wraps(_original_handle_invalid_blocks)
def _patched_handle_invalid_blocks(
    self: Scheduler,
    invalid_block_ids: set[int],
    num_scheduled_tokens: dict[str, int],
) -> set[str]:
    """Map Hybrid private-SWA load failures without recomputation.

    Upstream's failure mapper currently assumes a single KV group. Private
    SWA uses a hybrid layout, so the connector records one scheduler-visible
    shared group on each receiving request and reports IDs from that same
    group. A failed private-SWA import is never eligible for recomputation.
    """
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_swa_pool", None
    )
    if private_pool is None:
        return _original_handle_invalid_blocks(
            self, invalid_block_ids, num_scheduled_tokens
        )

    receiving_requests = [
        request
        for request in (*self.skipped_waiting, *self.running)
        if (
            getattr(request, "private_swa_transfer_state", None)
            == PRIVATE_SWA_TRANSFER_RECEIVING
        )
    ]
    failed_request_ids = {
        request.request_id
        for request in receiving_requests
        if invalid_block_ids.intersection(
            getattr(
                request,
                "private_swa_transfer_shared_block_ids",
                (),
            )
        )
    }
    if failed_request_ids:
        return failed_request_ids

    if receiving_requests:
        # A private-SWA load failure cannot be recovered safely. If a worker
        # reports IDs that cannot be mapped (for example due to a connector
        # version mismatch), fail all in-flight private imports rather than
        # accidentally promoting one with incomplete SWA.
        failed_request_ids = {
            request.request_id for request in receiving_requests
        }
        logger.error(
            "Private SWA load errors %s did not match request block IDs; "
            "failing all receiving private-SWA requests: %s",
            sorted(invalid_block_ids),
            sorted(failed_request_ids),
        )
        return failed_request_ids

    return _original_handle_invalid_blocks(
        self, invalid_block_ids, num_scheduled_tokens
    )


@wraps(_original_update_from_output)
def _patched_update_from_output(
    self: Scheduler,
    scheduler_output: SchedulerOutput,
    model_runner_output: Any,
) -> Any:
    for request_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        if request := self.requests.get(request_id):
            request.num_in_flight_tokens -= num_scheduled_tokens
            if request.num_in_flight_tokens < 0:
                _handle_negative_in_flight(
                    request_id,
                    request.num_in_flight_tokens,
                    "update_from_output",
                    num_scheduled_tokens,
                )
                request.num_in_flight_tokens = 0
    return _original_update_from_output(self, scheduler_output, model_runner_output)


_original_allocate_slots = KVCacheManager.allocate_slots


@wraps(_original_allocate_slots)
def _patched_allocate_slots(self: KVCacheManager, request: Request, *args: Any, **kwargs: Any) -> Any:
    token = _prune_context.set((request.request_id, _safe_in_flight_tokens(request, "allocate_slots")))
    private_pool: PrivateSWAAllocationPool | None = getattr(
        self.coordinator, "private_swa_pool", None
    )
    reserved_here = False
    if private_pool is not None and not private_pool.contains(request.request_id):
        allocation = private_pool.reserve(request.request_id)
        if allocation is None:
            _prune_context.reset(token)
            return None
        request.private_swa_allocation = allocation
        reserved_here = True
    try:
        result = _original_allocate_slots(self, request, *args, **kwargs)
        if result is None and reserved_here and private_pool is not None:
            private_pool.release(request.request_id)
            request.private_swa_allocation = None
        return result
    except Exception:
        if reserved_here and private_pool is not None:
            private_pool.release(request.request_id)
            request.private_swa_allocation = None
        raise
    finally:
        _prune_context.reset(token)


_original_kv_cache_manager_free = KVCacheManager.free


@wraps(_original_kv_cache_manager_free)
def _patched_kv_cache_manager_free(self: KVCacheManager, request: Request) -> None:
    try:
        _original_kv_cache_manager_free(self, request)
    finally:
        private_pool: PrivateSWAAllocationPool | None = getattr(
            self.coordinator, "private_swa_pool", None
        )
        if private_pool is not None:
            private_pool.release(request.request_id)
        request.private_swa_allocation = None
        request.private_swa_transfer_state = PRIVATE_SWA_TRANSFER_NONE
        request.private_swa_imported_window_start = None
        request.private_swa_imported_valid_length = None


_original_connector_finished = Scheduler._connector_finished


@wraps(_original_connector_finished)
def _patched_connector_finished(self: Scheduler, request: Request) -> tuple[bool, dict[str, Any] | None]:
    token = _prune_context.set((request.request_id, _safe_in_flight_tokens(request, "connector_finished")))
    try:
        result = _original_connector_finished(self, request)
        delay_free, _ = result
        private_pool = getattr(
            self.kv_cache_manager.coordinator, "private_swa_pool", None
        )
        allocation = getattr(request, "private_swa_allocation", None)
        if (
            delay_free
            and private_pool is not None
            and allocation is not None
            and request.private_swa_transfer_allocation is None
        ):
            private_pool.acquire_ref(int(allocation))
            request.private_swa_transfer_allocation = int(allocation)
        return result
    finally:
        _prune_context.reset(token)


_original_free_blocks = Scheduler._free_blocks


@wraps(_original_free_blocks)
def _patched_free_blocks(self: Scheduler, request: Request) -> None:
    allocation = getattr(request, "private_swa_transfer_allocation", None)
    private_pool = getattr(
        self.kv_cache_manager.coordinator, "private_swa_pool", None
    )
    try:
        _original_free_blocks(self, request)
    finally:
        if allocation is not None and private_pool is not None:
            private_pool.release_ref(int(allocation))
        request.private_swa_transfer_allocation = None


_original_remove_skipped_blocks = SingleTypeKVCacheManager.remove_skipped_blocks


@wraps(_original_remove_skipped_blocks)
def _patched_remove_skipped_blocks(
    self: SingleTypeKVCacheManager,
    request_id: str,
    total_computed_tokens: int,
) -> None:
    context = _prune_context.get()
    if (
        context is not None
        and context[0] == request_id
        and isinstance(self.kv_cache_spec, (ChunkedLocalAttentionSpec, SlidingWindowSpec))
    ):
        num_in_flight_tokens = context[1]
        if num_in_flight_tokens < 0:
            _handle_negative_in_flight(
                request_id,
                num_in_flight_tokens,
                "remove_skipped_blocks",
            )
            num_in_flight_tokens = 0
        total_computed_tokens = max(0, total_computed_tokens - num_in_flight_tokens)
    _original_remove_skipped_blocks(self, request_id, total_computed_tokens)


def _patched_chunked_local_max_memory_usage_bytes(self: ChunkedLocalAttentionSpec, vllm_config: VllmConfig) -> int:
    max_blocks = self.max_admission_blocks_per_request(
        max_num_batched_tokens=_max_in_flight_tokens(vllm_config),
        max_model_len=vllm_config.model_config.max_model_len,
    )
    return max_blocks * self.page_size_bytes


def _patched_swa_max_memory_usage_bytes(self: SlidingWindowSpec, vllm_config: VllmConfig) -> int:
    assert vllm_config.parallel_config.decode_context_parallel_size == 1, "DCP not support sliding window."
    max_blocks = self.max_admission_blocks_per_request(
        max_num_batched_tokens=_max_in_flight_tokens(vllm_config),
        max_model_len=vllm_config.model_config.max_model_len,
    )
    return max_blocks * self.page_size_bytes


_original_scheduler_init = Scheduler.__init__


@wraps(_original_scheduler_init)
def _patched_scheduler_init(self: Scheduler, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
    _original_scheduler_init(self, vllm_config, *args, **kwargs)
    coordinator = self.kv_cache_manager.coordinator
    private_group_ids = (list(getattr(coordinator, "private_swa_group_ids", ()))
                           if envs.VLLM_ASCEND_ENABLE_PRIVATE_SWA_POOL else [])
    if private_group_ids:
        private_spec = _find_private_swa_spec(
            coordinator.single_type_managers[private_group_ids[0]].kv_cache_spec)
        if private_spec is None:
            raise RuntimeError("DeepSeek-V4 private SWA group has no sliding-window spec")
        speculative_config = vllm_config.speculative_config
        draft_tokens = (
            getattr(speculative_config, "num_speculative_tokens", 0)
            if speculative_config is not None
            else 0
        )
        private_config = PrivateSWAConfig(
            block_size=private_spec.block_size,
            window_size=private_spec.sliding_window,
            in_flight_tokens=1 + draft_tokens,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
        )
        coordinator.private_swa_group_ids = frozenset(private_group_ids)
        coordinator.private_swa_config = private_config
        coordinator.private_swa_pool = PrivateSWAAllocationPool(private_config)
        # The D node must use the exact private SWA imported from P. Any load
        # failure is terminal because recomputing the prefix would recreate
        # the A3 overwrite bug this protocol is designed to prevent.
        self.recompute_kv_load_failures = False

    max_in_flight_tokens = _max_in_flight_tokens(vllm_config)
    for manager in coordinator.single_type_managers:
        spec = manager.kv_cache_spec
        if isinstance(spec, (ChunkedLocalAttentionSpec, SlidingWindowSpec)):
            manager._max_admission_blocks_per_request = spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_in_flight_tokens,
                max_model_len=self.max_model_len,
            )


Request.__init__ = _patched_request_init
Scheduler.__init__ = _patched_scheduler_init
Scheduler._update_after_schedule = _patched_update_after_schedule
Scheduler._handle_invalid_blocks = _patched_handle_invalid_blocks
Scheduler.update_from_output = _patched_update_from_output
Scheduler._connector_finished = _patched_connector_finished
Scheduler._free_blocks = _patched_free_blocks
KVCacheManager.allocate_slots = _patched_allocate_slots
KVCacheManager.free = _patched_kv_cache_manager_free
SingleTypeKVCacheManager.remove_skipped_blocks = _patched_remove_skipped_blocks
ChunkedLocalAttentionSpec.max_memory_usage_bytes = _patched_chunked_local_max_memory_usage_bytes
SlidingWindowSpec.max_memory_usage_bytes = _patched_swa_max_memory_usage_bytes
