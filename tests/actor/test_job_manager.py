# Copyright 2024 Cisco Systems, Inc. and its affiliates
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

import pytest
from infscale.actor.job_manager import JobManager
from tests.actor.conftest import job_config_diffs


@pytest.mark.parametrize(
    "old_config,new_config,expected_terminate_ids,expected_start_ids,expected_updated_ids",
    job_config_diffs,
)
def test_compare_configs(
    old_config,
    new_config,
    expected_terminate_ids,
    expected_start_ids,
    expected_updated_ids,
):
    job_mgr = JobManager()
    results = job_mgr.compare_configs(old_config, new_config)
    start_ids, updated_ids, terminate_ids = results

    assert set(start_ids) == set(expected_start_ids)
    assert set(updated_ids) == set(expected_updated_ids)
    assert set(terminate_ids) == set(expected_terminate_ids)
