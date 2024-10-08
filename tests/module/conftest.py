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

supported_model_names = [
    # language models
    "bert-base-uncased",
    "gpt2",
    "t5-small",
    # image models
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
]

datasets = [
    ("wikitext", "wikitext-2-raw-v1", "test"),
    ("tiny_shakespeare", "", "test"),
    ("wikitext", "wikitext-2-raw-v1", "test"),
    ("Maysee/tiny-imagenet", "", "valid"),
    ("cifar10", "", "test"),
]

model_dataset_pairs = list(zip(supported_model_names, datasets))
# old_config,new_config,expected_terminate_ids,expected_start_ids,expected_updated_ids
job_config_diffs = [
    # Test case 1: No changes
    (
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}],
            },
        ),
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}],
            },
        ),
        [],  # Expected terminate_ids
        [],  # Expected start_ids
        [],  # Expected updated_ids
    ),
    # # Test case 2: One worker updated, one started
    (
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}],
            },
        ),
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "0-1": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}, {"peers": ["0-1"]}],
            },
        ),
        [],  # Expected terminate_ids
        ["0-1"],  # Expected start_ids
        ["1-0"],  # Expected updated_ids
    ),
    # Test case 3: One worker terminated, one updated
    (
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "0-1": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}, {"peers": ["0-1"]}],
            },
        ),
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}],
            },
        ),
        ["0-1"],  # Expected terminate_ids
        [],  # Expected start_ids
        ["1-0"],  # Expected updated_ids
    ),
    # # Test case 4: All workers updated
    (
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-0": [{"peers": ["1-0"]}],
                "0-0": [{"peers": ["s-0"]}],
                "1-0": [{"peers": ["0-0"]}],
            },
        ),
        JobConfig(
            workers=[],
            name="test",
            model="model",
            rank_map={},
            dataset=None,
            flow_graph={
                "s-4": [{"peers": ["4-0"]}],
                "2-0": [{"peers": ["s-4"]}],
                "4-0": [{"peers": ["2-0"]}],
            },
        ),
        ["s-0", "0-0", "1-0"],  # Expected terminate_ids
        ["s-4", "2-0", "4-0"],  # Expected start_ids
        [],  # Expected updated_ids
    ),
]
