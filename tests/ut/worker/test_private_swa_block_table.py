from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.block_table import MultiGroupBlockTable


class _FakeBlockTable:
    def __init__(self, *, is_private: bool, rows: int = 2, columns: int = 64):
        self.is_private_swa_group = is_private
        self.is_mamba_group = False
        self.physical_block_size = 32
        self.block_table = SimpleNamespace(
            np=np.zeros((rows, columns), dtype=np.int32)
        )
        self.slot_mapping = SimpleNamespace(
            gpu=torch.arange(8, dtype=torch.int32)
        )

    def clear_row(self, row: int) -> None:
        self.block_table.np[row].fill(0)

    def add_row(self, block_ids: list[int], row: int) -> None:
        self.clear_row(row)
        self.block_table.np[row, : len(block_ids)] = block_ids


def _new_multi_group_table(
    *block_tables: _FakeBlockTable,
) -> MultiGroupBlockTable:
    table = MultiGroupBlockTable.__new__(MultiGroupBlockTable)
    table.block_tables = list(block_tables)
    table.private_swa_allocation_ids = torch.full(
        (2,), -1, dtype=torch.int64, device="cpu"
    )
    table.private_swa_blocks_per_allocation = 0
    table.private_swa_id_codec = None
    return table


def test_private_swa_layout_accepts_highest_pool_handle():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    table.set_private_swa_allocations(
        allocation_ids=np.array([7], dtype=np.int64),
        blocks_per_allocation=5,
        num_allocations=8,
        private_num_blocks=41,
        window_starts=np.array([132], dtype=np.int64),
        valid_lengths=np.array([260], dtype=np.int64),
    )

    assert table.private_swa_id_codec is not None
    assert table.private_swa_id_codec.private_num_blocks == 41
    assert private.block_table.np[0, 4] == 40

    with pytest.raises(IndexError, match="handle is out of range"):
        table.set_private_swa_allocations(
            allocation_ids=np.array([8], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=41,
        )


def test_private_swa_layout_rejects_inconsistent_pool_capacity():
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(private)

    with pytest.raises(ValueError, match="block count does not match"):
        table.set_private_swa_allocations(
            allocation_ids=np.array([0], dtype=np.int64),
            blocks_per_allocation=5,
            num_allocations=8,
            private_num_blocks=21,
        )


def test_rollback_masks_shared_slots_but_keeps_private_ring_slots():
    shared = _FakeBlockTable(is_private=False)
    private = _FakeBlockTable(is_private=True)
    table = _new_multi_group_table(shared, private)
    positions = torch.tensor([127, 128, 200, 31], dtype=torch.int64)
    request_indices = torch.tensor([0, 0, 0, 1], dtype=torch.int64)
    persistent_starts = torch.tensor([128, -1], dtype=torch.int64)
    private_before = private.slot_mapping.gpu.clone()

    table.mask_shared_slots_for_private_swa_rollback(
        positions,
        request_indices,
        persistent_starts,
    )

    assert shared.slot_mapping.gpu[:4].tolist() == [-1, 1, 2, 3]
    assert torch.equal(private.slot_mapping.gpu, private_before)
