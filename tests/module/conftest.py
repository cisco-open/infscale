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
