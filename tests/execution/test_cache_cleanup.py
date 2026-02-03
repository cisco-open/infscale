# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Test EMA-based adaptive cache cleanup mechanism."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import DynamicCache

from infscale.execution.stage import Stage
from infscale.module.model_metadata import Llama3ModelMetaData
from infscale.module.modelir import ModelIR


@pytest.fixture
def mock_modelir():
    """Create a mock ModelIR with Llama3 metadata."""
    modelir = Mock(spec=ModelIR)
    modelir.mmd = Mock(spec=Llama3ModelMetaData)
    modelir.layers = [Mock() for _ in range(10)]
    modelir.output_parser = None
    return modelir


@pytest.fixture
def stage(mock_modelir):
    """Create a Stage instance for testing."""
    with patch.object(Stage, "_init_layers"):
        stage = Stage(
            stage_id="test-stage",
            modelir=mock_modelir,
            start=0,
            end=9,
            device=torch.device("cpu"),
            max_inflight=10,
        )
    return stage


def test_cache_initialization(stage):
    """Test that cache structure is initialized correctly."""
    assert isinstance(stage.caches, dict)
    assert len(stage.caches) == 0
    assert stage.ema_inter_arrival is None
    assert stage.last_request_time is None
    assert stage.adaptive_timeout == 10.0
    assert stage.ema_alpha == 0.5
    assert stage.timeout_multiplier == 10
    assert stage.min_timeout == 0.5
    assert stage.max_timeout == 30.0


def test_update_adaptive_timeout_first_request(stage):
    """Test adaptive timeout update on first request."""
    # First request
    stage._update_adaptive_timeout()
    
    assert stage.last_request_time is not None
    assert stage.ema_inter_arrival is None  # No inter-arrival yet
    assert stage.adaptive_timeout == 10.0  # Still default


def test_update_adaptive_timeout_second_request(stage):
    """Test adaptive timeout update on second request."""
    # First request
    stage._update_adaptive_timeout()
    time.sleep(0.1)
    
    # Second request
    stage._update_adaptive_timeout()
    
    assert stage.ema_inter_arrival is not None
    assert stage.ema_inter_arrival > 0.09  # Should be ~0.1s
    # Timeout = 0.1s * 2 * 10 = 2s (bounded by min/max)
    assert stage.adaptive_timeout >= stage.min_timeout
    assert stage.adaptive_timeout <= stage.max_timeout


def test_cache_creation_with_timestamp(stage):
    """Test that caches are created with timestamps."""
    # Mock _run_llm
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # First request
    outputs, next_layer = stage._llm_generate(100, input_ids=torch.tensor([[1, 2, 3]]))
    
    assert 100 in stage.caches
    cache, timestamp = stage.caches[100]
    assert isinstance(cache, DynamicCache)
    assert isinstance(timestamp, float)
    assert timestamp > 0


def test_cache_timestamp_update(stage):
    """Test that cache timestamps are updated on access."""
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # First access
    stage._llm_generate(100, input_ids=torch.tensor([[1, 2, 3]]))
    _, first_timestamp = stage.caches[100]
    
    time.sleep(0.05)
    
    # Second access
    stage._llm_generate(100, input_ids=torch.tensor([[4, 5, 6]]))
    _, second_timestamp = stage.caches[100]
    
    assert second_timestamp > first_timestamp


def test_cleanup_idle_caches_no_eviction(stage):
    """Test that active caches are not evicted."""
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # Create a cache
    stage._llm_generate(100, input_ids=torch.tensor([[1, 2, 3]]))
    
    # Immediate cleanup - should not evict
    stage._cleanup_idle_caches()
    
    assert 100 in stage.caches


def test_cleanup_idle_caches_with_eviction(stage):
    """Test that idle caches are evicted after timeout."""
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # Create a cache
    stage._llm_generate(100, input_ids=torch.tensor([[1, 2, 3]]))
    
    # Manually set an old timestamp to simulate idle cache
    cache, _ = stage.caches[100]
    old_timestamp = time.time() - 15.0  # 15 seconds ago
    stage.caches[100] = (cache, old_timestamp)
    
    # Force cleanup to run (bypass rate limiting)
    stage.last_cleanup_time = 0
    
    # Run cleanup - should evict
    stage._cleanup_idle_caches()
    
    assert 100 not in stage.caches


def test_out_of_order_completion(stage):
    """Test that out-of-order completion is handled correctly."""
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # Simulate three requests arriving
    stage._llm_generate(100, input_ids=torch.tensor([[1]]))  # Slow (will take 64 tokens)
    time.sleep(0.05)
    stage._llm_generate(101, input_ids=torch.tensor([[2]]))  # Fast (will take 10 tokens)
    time.sleep(0.05)
    stage._llm_generate(102, input_ids=torch.tensor([[3]]))  # Medium (will take 30 tokens)
    
    # All three should be in cache
    assert 100 in stage.caches
    assert 101 in stage.caches
    assert 102 in stage.caches
    
    # Simulate seqno=101 completing first (out of order)
    # Mark it as old by setting an old timestamp
    cache_101, _ = stage.caches[101]
    stage.caches[101] = (cache_101, time.time() - 15.0)
    
    # Force cleanup
    stage.last_cleanup_time = 0
    stage._cleanup_idle_caches()
    
    # Only 101 should be evicted, 100 and 102 should remain
    assert 100 in stage.caches
    assert 101 not in stage.caches
    assert 102 in stage.caches


def test_adaptive_timeout_calculation(stage):
    """Test that adaptive timeout scales with inter-arrival time and max_inflight."""
    # Simulate fast inter-arrival (0.1s)
    stage._update_adaptive_timeout()
    time.sleep(0.1)
    stage._update_adaptive_timeout()
    
    # After a few updates, EMA should stabilize around 0.1s
    for _ in range(5):
        time.sleep(0.1)
        stage._update_adaptive_timeout()
    
    # Timeout should be: ~0.1s * 2 * 10 = ~2s
    expected_timeout = stage.ema_inter_arrival * stage.timeout_multiplier * stage.max_inflight
    assert abs(stage.adaptive_timeout - expected_timeout) < 0.1
    
    # Verify it's bounded
    assert stage.adaptive_timeout >= stage.min_timeout
    assert stage.adaptive_timeout <= stage.max_timeout


def test_ema_smoothing(stage):
    """Test that EMA smooths out inter-arrival variations."""
    stage._update_adaptive_timeout()
    
    # Simulate varying inter-arrival times
    time.sleep(0.1)
    stage._update_adaptive_timeout()
    
    first_ema = stage.ema_inter_arrival
    
    time.sleep(0.2)  # Longer interval
    stage._update_adaptive_timeout()
    
    # EMA should have increased but not doubled (due to smoothing)
    assert stage.ema_inter_arrival > first_ema
    assert stage.ema_inter_arrival < first_ema * 2


def test_no_cache_leak_with_replicas(stage):
    """Test that replicas don't cause memory leaks (seqnos can be non-contiguous)."""
    stage._run_llm = Mock(return_value={"output": torch.tensor([1.0])})
    
    # Simulate replica A getting only even seqnos
    for seqno in [0, 2, 4, 6, 8, 10]:
        stage._llm_generate(seqno, input_ids=torch.tensor([[seqno]]))
        time.sleep(0.01)
    
    # Mark old seqnos as completed (set old timestamps)
    for seqno in [0, 2, 4]:
        cache, _ = stage.caches[seqno]
        stage.caches[seqno] = (cache, time.time() - 15.0)
    
    # Force cleanup
    stage.last_cleanup_time = 0
    stage._cleanup_idle_caches()
    
    # Old seqnos should be evicted
    assert 0 not in stage.caches
    assert 2 not in stage.caches
    assert 4 not in stage.caches
    
    # Recent seqnos should remain
    assert 6 in stage.caches
    assert 8 in stage.caches
    assert 10 in stage.caches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
