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

"""Stage class."""

from __future__ import annotations

import traceback
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from infscale import get_logger
from torch.nn import Parameter

if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


logger = get_logger()


class Stage(nn.Module):
    """Stage class."""

    def __init__(
        self,
        stage_id: str,
        layers: list[fx.GraphModule],
        device: torch.device = torch.device("cpu"),
        output_parser: Callable = None,
        modelir=None,
    ):
        """Initialize stage class instance."""
        super().__init__()

        self.id = stage_id
        self.layers = deepcopy(layers)
        self.device = device

        self.modelir = modelir

        try:
            self._init_layers()
        except Exception as e:
            traceback.print_exc()
            raise e

        # An output parser is only useful for the last stage.
        # The outputs from the last stage need to be sent back to the inference
        # server. Therefore they need to be sent back as a list of tensors.
        # But if the output is a dictionary of tensors. This leads to comm
        # error. Also, in the inference, other values such as loss may not be
        # important. So, a way to manipulate the outputs is provided.
        self._output_parser: Union[Callable, None] = output_parser

    def forward(self, **inputs) -> dict[str, Tensor]:
        """Run layers in the stage."""
        for layer in self.layers:
            inputs = layer(**inputs)

        outputs = self._output_parser(inputs) if self._output_parser else inputs

        return outputs

    def _init_layers(self):
        """Initialize meta layers and move them to a device."""
        model = self.modelir.mmd.load_model()

        named_parameters = dict()
        for name, param in model.named_parameters():
            named_parameters[name] = param

        # Huggingface's CausalLM models don't include lm_head as model parameter
        # see https://github.com/huggingface/transformers/issues/6291
        # but init_empty_weights() somehow includes lm_head as model parameter
        # To initialize layers correctly, we include lm_head as well
        # Not sure if this is a correct way to handle the issue
        if hasattr(model, "lm_head"):
            for name, param in model.lm_head.named_parameters():
                name = "lm_head." + name
                named_parameters[name] = param

        named_buffers = dict()
        for name, buffer in model.named_buffers():
            named_buffers[name] = buffer

        for layer in self.layers:
            self._init_tensors(layer, named_parameters, named_buffers)

        del named_parameters
        del named_buffers
        del model

    def _init_tensors(
        self,
        layer: torch.fx.GraphModule,
        named_parameters: dict[str, Parameter],
        named_buffers: dict[str, Tensor],
    ):
        """Initialize meta tensors and move them to a device."""
        for name, _ in layer.named_parameters():
            assert name in named_parameters, f"parameter {name} not found"

            set_module_tensor_to_device(
                layer,
                name,
                self.device,
                named_parameters[name].data,
            )

        for name, _ in layer.named_buffers():
            assert name in named_buffers, f"buffer {name} not found"

            set_module_tensor_to_device(
                layer,
                name,
                self.device,
                named_buffers[name].data,
            )
