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
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.sharder import Sharder
from infscale.module.zoo import Zoo
from tests.module.conftest import model_dataset_pairs


@pytest.mark.parametrize("model_name,dataset_info", model_dataset_pairs)
def test_shard(model_name, dataset_info):
    mmd = Zoo.get_model_metadata(model_name)
    assert mmd is not None

    dataset_path, dataset_name, split = dataset_info
    dataset = HuggingFaceDataset(mmd, dataset_path, dataset_name, split)
    assert dataset.sample

    layers = Sharder.shard(mmd, list(dataset.sample.keys()))
    assert layers is not None and len(layers) > 0
