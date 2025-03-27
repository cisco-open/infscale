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

from infscale.module.zoo import Zoo
from tests.module.conftest import supported_model_names


bad_testdata = [
    ("albert-base-v1", KeyError),  # not supported
    ("noexist_model", OSError),  # doesn't exist
]


@pytest.mark.parametrize("model_name", supported_model_names)
def test_get_model_metadata_success(model_name):
    model_md = Zoo.get_model_metadata(model_name)
    assert model_md is not None


@pytest.mark.parametrize("model_name,expected", bad_testdata)
def test_get_model_metadata_fail(model_name, expected):
    with pytest.raises((EnvironmentError, KeyError)) as excinfo:
        _ = Zoo.get_model_metadata(model_name)
    assert excinfo.type is expected
