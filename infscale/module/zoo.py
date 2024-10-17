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

"""Zoo class."""

from infscale.module.model_metadata import (BaseModelMetaData,
                                            BertModelMetaData,
                                            Gpt2ModelMetaData,
                                            Llama3ModelMetaData,
                                            ResnetModelMetaData,
                                            T5ModelMetaData, VitModelMetaData)
from transformers import AutoConfig


class Zoo:
    """Collection of models supported in InfScale."""

    model_metadata_dict = {
        "llama": Llama3ModelMetaData,
        "openai-gpt": Gpt2ModelMetaData,
        "gpt2": Gpt2ModelMetaData,
        "bert": BertModelMetaData,
        "t5": T5ModelMetaData,
        "vit": VitModelMetaData,
        "resnet": ResnetModelMetaData,
    }

    @classmethod
    def get_model_metadata(cls, name: str) -> BaseModelMetaData:
        """Return a meta model."""
        config = AutoConfig.from_pretrained(name)

        model_type = config.model_type
        if model_type not in cls.model_metadata_dict:
            raise KeyError(f"Model type '{model_type}' is not supported.")

        return cls.model_metadata_dict[model_type](name, config)
