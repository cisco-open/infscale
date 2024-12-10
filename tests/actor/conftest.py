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

from infscale.config import JobConfig

# old_config,new_config,expected_terminate_ids,expected_start_ids,expected_updated_ids
job_config_diffs = [
    # Test case 1: No changes
    (
        JobConfig(
            job_id="job1",
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["0-0"],
                    }
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["0-0"],
                    }
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
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["0-0"],
                    }
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "0-1": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w3",
                        "addr": "127.0.0.1",
                        "port": 32500,
                        "peers": ["0-0"],
                    },
                    {
                        "name": "w4",
                        "addr": "127.0.0.1",
                        "port": 33500,
                        "peers": ["0-1"],
                    }],
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
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "0-1": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w3",
                        "addr": "127.0.0.1",
                        "port": 32500,
                        "peers": ["0-0"],
                    },
                    {
                        "name": "w4",
                        "addr": "127.0.0.1",
                        "port": 33500,
                        "peers": ["0-1"],
                    }],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["0-0"],
                    }
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
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-0": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["1-0"],
                    }
                ],
                "0-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-0"],
                    }
                ],
                "1-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["0-0"],
                    }
                ],
            },
        ),
        JobConfig(
            job_id="job1",
            workers=[],
            name="test",
            model="model",
            dataset=None,
            flow_graph={
                "s-4": [
                    {
                        "name": "w0",
                        "addr": "127.0.0.1",
                        "port": 29500,
                        "peers": ["4-0"],
                    }
                ],
                "2-0": [
                    {
                        "name": "w1",
                        "addr": "127.0.0.1",
                        "port": 30500,
                        "peers": ["s-4"],
                    }
                ],
                "4-0": [
                    {
                        "name": "w2",
                        "addr": "127.0.0.1",
                        "port": 31500,
                        "peers": ["2-0"],
                    }
                ],
            },
        ),
        ["s-0", "0-0", "1-0"],  # Expected terminate_ids
        ["s-4", "2-0", "4-0"],  # Expected start_ids
        [],  # Expected updated_ids
    ),
]
