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

"""conftest file."""

from infscale.configs.job import JobConfig, WorkerData, WorldInfo


# old_config,new_config,expected_terminate_ids,expected_start_ids,expected_updated_ids
job_config_diffs = [
    # Test case 1: No changes
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        [],  # Expected terminate_ids
        [],  # Expected start_ids
        [],  # Expected updated_ids
    ),
    # # Test case 2: Two workers updated, one started
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-1", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-1": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w3",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    ),
                    WorldInfo(
                        **{
                            "name": "w4",
                            "peers": ["0-1"],
                            "backend": "gloo",
                        }
                    ),
                ],
            },
        ),
        [],  # Expected terminate_ids
        ["0-1"],  # Expected start_ids
        ["1-0", "s-0"],  # Expected updated_ids
    ),
    # Test case 3: One worker terminated, two updated
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-1", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-1": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w3",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    ),
                    WorldInfo(
                        **{
                            "name": "w4",
                            "peers": ["0-1"],
                            "backend": "gloo",
                        }
                    ),
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        ["0-1"],  # Expected terminate_ids
        [],  # Expected start_ids
        ["1-0", "s-0"],  # Expected updated_ids
    ),
    # # Test case 4: All workers updated
    (
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "0-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "1-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["1-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "0-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "1-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["0-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[
                WorkerData(**{"id": "s-4", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "2-0", "stage": {}, "device": "cpu"}),
                WorkerData(**{"id": "4-0", "stage": {}, "device": "cpu"}),
            ],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-4": [
                    WorldInfo(
                        **{
                            "name": "w0",
                            "peers": ["4-0"],
                            "backend": "gloo",
                        }
                    )
                ],
                "2-0": [
                    WorldInfo(
                        **{
                            "name": "w1",
                            "peers": ["s-4"],
                            "backend": "gloo",
                        }
                    )
                ],
                "4-0": [
                    WorldInfo(
                        **{
                            "name": "w2",
                            "peers": ["2-0"],
                            "backend": "gloo",
                        }
                    )
                ],
            },
        ),
        ["s-0", "0-0", "1-0"],  # Expected terminate_ids
        ["s-4", "2-0", "4-0"],  # Expected start_ids
        [],  # Expected updated_ids
    ),
]
