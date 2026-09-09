# SPDX-License-Identifier: Apache-2.0
"""Request-private SWA allocation bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass

from threading import Lock
from typing import Literal

from vllm.logger import logger

PoolType = Literal["shared", "private"]

PRIVATE_SWA_TRANSFER_NONE = "none"
PRIVATE_SWA_TRANSFER_RECEIVING = "receiving"
PRIVATE_SWA_TRANSFER_READY = "ready"
PRIVATE_SWA_TRANSFER_FAILED = "failed"


def is_private_swa_kv_cache_spec(spec: object) -> bool:
    """Return whether a leaf KV cache spec belongs to the DSV4 SWA pool."""
    return (
        getattr(spec, "model_version", None) == "deepseek_v4"
        and getattr(spec, "sliding_window", None) is not None
    )


@dataclass(frozen=True)
class TypedBlockId:
    """Host-side pool-aware block identifier."""

    pool_type: PoolType
    local_id: int


@dataclass(frozen=True)
class PrivateSWAIdCodec:
    """Encode typed scheduler block IDs at the tensor/kernel boundary."""

    shared_num_blocks: int
    private_num_blocks: int

    def __post_init__(self) -> None:
        if self.shared_num_blocks < 0 or self.private_num_blocks < 0:
            raise ValueError("block counts must be non-negative")

    @property
    def private_base(self) -> int:
        return self.shared_num_blocks + 1

    def encode(self, pool_type: PoolType, local_block_id: int) -> int:
        if local_block_id < 0:
            raise ValueError("local block id must be non-negative")
        if pool_type == "shared":
            if local_block_id >= self.shared_num_blocks:
                raise IndexError("shared block id out of range")
            return local_block_id + 1
        if pool_type == "private":
            if local_block_id >= self.private_num_blocks:
                raise IndexError("private block id out of range")
            return self.private_base + local_block_id
        raise ValueError(f"unknown pool type: {pool_type}")

    def decode_for_kernel(self, encoded_id: int, expected_pool: PoolType) -> int:
        """Decode a host ID to a local index for an unchanged kernel."""
        decoded = self.decode(encoded_id)
        if decoded is None:
            return 0
        pool_type, local_id = decoded
        if pool_type != expected_pool:
            raise ValueError(f"block pool mismatch: expected {expected_pool}, got {pool_type}")
        return local_id

    def decode(self, encoded_id: int) -> tuple[PoolType, int] | None:
        if encoded_id == 0:
            return None
        if encoded_id < 0:
            raise ValueError("encoded block id must be non-negative")
        if 1 <= encoded_id <= self.shared_num_blocks:
            return "shared", encoded_id - 1
        local = encoded_id - self.private_base
        if 0 <= local < self.private_num_blocks:
            return "private", local
        raise IndexError("encoded block id out of range")



def compute_prefix_rollback_start(
    hit_length: int,
    rollback_tokens: int,
) -> int:
    """Return the exact request-private SWA recompute start.

    Shared cache accounting stays at ``hit_length``. The returned position is
    only used to prepend private-SWA warm-up tokens in the model runner, so it
    must not be aligned to a shared-cache block size.
    """
    if hit_length < 0 or rollback_tokens < 0:
        raise ValueError("hit_length and rollback_tokens must be non-negative")
    return max(0, hit_length - rollback_tokens)


def detect_new_prefix_hit_length(
    computed_before_schedule: int,
    computed_after_schedule: int,
    num_scheduled_tokens: int,
    waiting_for_remote_kvs: bool = False,
) -> int | None:
    """Detect a first-step prefix hit from scheduler-visible positions.

    Local cache hits are normally recorded at ``get_computed_blocks``. This
    fallback covers external KV connectors, including asynchronous loads that
    advance ``num_computed_tokens`` without scheduling model tokens.
    """
    if min(
        computed_before_schedule,
        computed_after_schedule,
        num_scheduled_tokens,
    ) < 0:
        raise ValueError("scheduler token counts must be non-negative")
    if computed_before_schedule > 0 and not waiting_for_remote_kvs:
        return None
    confirmed_length = max(0, computed_after_schedule - num_scheduled_tokens)
    return confirmed_length or None


def compute_private_swa_prefill_workspace_blocks(
    max_num_batched_tokens: int,
    max_num_seqs: int,
    window_size: int,
    block_size: int,
) -> int:
    """Return the worst-case shared Prefill workspace size in blocks.

    A request needs either up to ``window_size - 1`` restored history tokens
    or up to ``window_size`` rollback tokens. Rollback rows restore no history,
    so the larger transient range is reserved once, plus page-alignment slack
    for every request.
    """
    if max_num_batched_tokens < 0:
        raise ValueError("max_num_batched_tokens must be non-negative")
    if max_num_seqs <= 0 or window_size <= 0 or block_size <= 0:
        raise ValueError(
            "max_num_seqs, window_size, and block_size must be positive"
        )
    max_workspace_tokens = (
        max_num_batched_tokens
        + max_num_seqs * (window_size + block_size - 1)
    )
    return (max_workspace_tokens + block_size - 1) // block_size


@dataclass(frozen=True)
class PrivateSWAConfig:
    """Sizing parameters for the request-private SWA pool.

    Compact ring sizing for A5. Repeated physical IDs are masked at token/offset granularity.
    """

    block_size: int
    window_size: int
    in_flight_tokens: int
    max_num_seqs: int
    delayed_free_slots: int | None = None

    @property
    def capacity_tokens(self) -> int:
        return self.window_size - 1 + self.in_flight_tokens

    @property
    def blocks_per_allocation(self) -> int:
        capacity = self.capacity_tokens
        return (capacity + self.block_size - 1) // self.block_size

    @property
    def num_allocations(self) -> int:
        delayed = self.max_num_seqs if self.delayed_free_slots is None else self.delayed_free_slots
        return self.max_num_seqs + delayed

    @property
    def num_blocks(self) -> int:
        return 1 + self.num_allocations * self.blocks_per_allocation




class PrivateSWAAllocationPool:
    """Small, thread-safe allocation ledger for private SWA slots.

    Physical KV tensors are owned by the KV-cache allocator; this class only
    owns request-to-allocation bookkeeping and is intentionally independent of
    prefix-cache block hashes.
    """

    def __init__(self, config: PrivateSWAConfig):
        if (config.block_size <= 0
                or config.window_size <= 0
                or config.in_flight_tokens <= 0
                or config.max_num_seqs <= 0):
            raise ValueError("private SWA sizing values must be positive")
        self.config = config
        if config.delayed_free_slots is not None and config.delayed_free_slots < 0:
            raise ValueError("delayed_free_slots must be non-negative")
        self._free = list(range(config.num_allocations))
        self._owned: dict[str, int] = {}
        # Key retained entries by allocation, not request. A request id may
        # be admitted again while a previous P/D transfer still owns its old
        # allocation; request-keyed storage would overwrite and leak it.
        self._retained: dict[int, str] = {}
        self._refs: dict[int, int] = {}
        self._lock = Lock()
        logger.info(
            "PRIVATE_SWA_POOL init block_size=%d window_size=%d in_flight=%d "
            "blocks_per_allocation=%d allocations=%d total_blocks=%d",
            config.block_size,
            config.window_size,
            config.in_flight_tokens,
            config.blocks_per_allocation,
            config.num_allocations,
            config.num_blocks,
        )

    def reserve(self, request_id: str) -> int | None:
        with self._lock:
            if request_id in self._owned:
                allocation = self._owned[request_id]
                logger.info(
                    "PRIVATE_SWA_POOL reserve_reuse request_id=%s allocation=%d "
                    "blocks=%d-%d free_allocations=%d",
                    request_id,
                    allocation,
                    self._first_block_id(allocation),
                    self._last_block_id(allocation),
                    len(self._free),
                )
                return allocation
            if not self._free:
                logger.info(
                    "PRIVATE_SWA_POOL reserve_exhausted request_id=%s "
                    "allocations=%d retained=%d",
                    request_id,
                    self.config.num_allocations,
                    len(self._retained),
                )
                return None
            allocation = self._free.pop()
            self._owned[request_id] = allocation
            logger.info(
                "PRIVATE_SWA_POOL reserve request_id=%s allocation=%d "
                "blocks=%d-%d free_allocations=%d",
                request_id,
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                len(self._free),
            )
            return allocation

    def release(self, request_id: str) -> bool:
        with self._lock:
            allocation = self._owned.pop(request_id, None)
            if allocation is None:
                logger.info("PRIVATE_SWA_POOL release_missing request_id=%s", request_id)
                return False
            refs = self._refs.get(allocation, 0)
            retained = refs > 0
            if retained:
                self._retained[allocation] = request_id
            else:
                self._free.append(allocation)
            logger.info(
                "PRIVATE_SWA_POOL release request_id=%s allocation=%d "
                "blocks=%d-%d refs=%d retained=%s free_allocations=%d",
                request_id,
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                refs,
                retained,
                len(self._free),
            )
            return True

    def acquire_ref(self, allocation: int) -> None:
        with self._lock:
            self._validate_allocation(allocation)
            refs = self._refs.get(allocation, 0) + 1
            self._refs[allocation] = refs
            logger.info(
                "PRIVATE_SWA_POOL acquire_ref allocation=%d blocks=%d-%d refs=%d",
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                refs,
            )

    def release_ref(self, allocation: int) -> bool:
        with self._lock:
            self._validate_allocation(allocation)
            refs = self._refs.get(allocation, 0)
            if refs <= 0:
                logger.info(
                    "PRIVATE_SWA_POOL release_ref_missing allocation=%d", allocation
                )
                return False
            released_to_free = False
            if refs == 1:
                self._refs.pop(allocation, None)
                if allocation in self._retained:
                    self._retained.pop(allocation)
                    self._free.append(allocation)
                    released_to_free = True
                refs = 0
            else:
                refs -= 1
                self._refs[allocation] = refs
            logger.info(
                "PRIVATE_SWA_POOL release_ref allocation=%d blocks=%d-%d refs=%d "
                "released_to_free=%s free_allocations=%d",
                allocation,
                self._first_block_id(allocation),
                self._last_block_id(allocation),
                refs,
                released_to_free,
                len(self._free),
            )
            return True

    def _first_block_id(self, allocation: int) -> int:
        return 1 + allocation * self.config.blocks_per_allocation

    def _last_block_id(self, allocation: int) -> int:
        return self._first_block_id(allocation) + self.config.blocks_per_allocation - 1

    def allocation_block_ids(self, allocation: int) -> range:
        self._validate_allocation(allocation)
        start = self._first_block_id(allocation)
        return range(start, start + self.config.blocks_per_allocation)

    def slot_for_position(self, allocation: int, absolute_position: int) -> int:
        """Return the physical slot for an absolute token position."""
        if absolute_position < 0:
            raise ValueError("absolute position must be non-negative")
        block_ids = self.allocation_block_ids(allocation)
        ring_block = (absolute_position // self.config.block_size) % len(block_ids)
        return block_ids[ring_block] * self.config.block_size + absolute_position % self.config.block_size

    def get_allocation(self, request_id: str) -> int | None:
        with self._lock:
            return self._owned.get(request_id)

    def request_for_allocation(self, allocation: int) -> str | None:
        """Return the live or retained owner of an allocation."""
        with self._lock:
            for request_id, owned in self._owned.items():
                if owned == allocation:
                    return request_id
            return self._retained.get(allocation)

    def ref_count(self, allocation: int) -> int:
        with self._lock:
            self._validate_allocation(allocation)
            return self._refs.get(allocation, 0)

    def contains(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._owned

    @property
    def free_allocations(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def retained_allocations(self) -> int:
        with self._lock:
            return len(self._retained)

    @property
    def used_allocations(self) -> int:
        with self._lock:
            return self.config.num_allocations - len(self._free)

    def over_high_watermark(self, threshold: float = 0.9) -> bool:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        return (self.used_allocations
                / self.config.num_allocations
                >= threshold)

    def _validate_allocation(self, allocation: int) -> None:
        if allocation < 0 or allocation >= self.config.num_allocations:
            raise IndexError("invalid private SWA allocation")
