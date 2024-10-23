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

from typing import List, Tuple

from infscale.config import JobConfig, WorkerInfo


def get_config_diff_ids(
    job_config: JobConfig, new_job_config: JobConfig
) -> Tuple[List[int], List[int], List[int]]:
    """Compares two flow_graph dictionaries, and returns diffs"""

    old_cfg = set(job_config.flow_graph.keys())
    new_cfg = set(new_job_config.flow_graph.keys())

    terminate_ids = list(old_cfg - new_cfg)
    start_ids = list(new_cfg - old_cfg)

    updated_ids = []
    for key in old_cfg & new_cfg:
        old_value = job_config.flow_graph[key]
        new_value = new_job_config.flow_graph[key]

        if len(old_value) != len(new_value):
            updated_ids.append(key)
            continue

        for old_worker, new_worker in zip(old_value, new_value):
            # TODO - remove isinstance check when the config file is being sent through the api call
            if old_worker.peers != (
                new_worker.peers
                if isinstance(new_worker, WorkerInfo)
                else new_worker["peers"]
            ):
                updated_ids.append(key)
                break

    return terminate_ids, start_ids, updated_ids
