# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1


def test_multi_group_recompute_policy_is_rejected():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    scheduler.recompute_kv_load_failures = True

    try:
        scheduler._validate_multi_group_kv_load_failure_policy()
    except ValueError as error:
        assert "kv_load_failure_policy='fail'" in str(error)
    else:
        raise AssertionError("Expected multi-group recompute policy to be rejected")


def test_multi_group_invalid_blocks_fail_sync_and_async_requests():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = MagicMock()

    sync_request = SimpleNamespace(
        request_id="sync-request",
        status=RequestStatus.RUNNING,
        num_computed_tokens=64,
    )
    async_request = SimpleNamespace(
        request_id="async-request",
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        num_computed_tokens=128,
    )
    unrelated_request = SimpleNamespace(
        request_id="unrelated-request",
        status=RequestStatus.RUNNING,
        num_computed_tokens=32,
    )
    scheduler.running = [sync_request, unrelated_request]
    scheduler.skipped_waiting = [async_request]

    block_ids_by_request = {
        "sync-request": ([1, 2], [10, 11, 12]),
        "async-request": ([20], [21, 22]),
        "unrelated-request": ([30], [31]),
    }
    scheduler.kv_cache_manager.get_block_ids.side_effect = block_ids_by_request.__getitem__

    affected_request_ids = scheduler._handle_invalid_blocks({11, 22}, {})

    assert affected_request_ids == {"sync-request", "async-request"}
    assert sync_request.num_computed_tokens == 64
    assert async_request.num_computed_tokens == 128
    scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({11, 12})
