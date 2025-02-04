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
from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from infscale import get_logger
from infscale.module.model_metadata import Llama3ModelMetaData
from infscale.module.modelir import ModelIR
from torch.nn import Parameter
from transformers import DynamicCache

if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


logger = None


class Stage(nn.Module):
    """Stage class."""

    def __init__(
        self,
        stage_id: str,
        modelir: ModelIR,
        start: int,
        end: int,
        device: torch.device = torch.device("cpu"),
        max_inflight: int = 1,
    ):
        """Initialize stage class instance."""
        super().__init__()
        global logger
        logger = get_logger()

        self.id = stage_id

        self.modelir = modelir

        self.start = start
        self.end = end

        self.device = device

        self.max_inflight = max_inflight

        # decide if this stage contains the first layer of a model
        self.is_first = start == 0
        # decide if this stage contains the last layer of a model
        self.is_last = end + 1 == len(modelir.layers)
        # decide if a full model is loaded
        # end + 1 - start == len(modelir.layers)
        self.is_full_model = self.is_first and self.is_last

        # resize the model layers so that other unused layers can be
        # garbage collected; not sure when/whether it happens though
        modelir.layers = modelir.layers[start : end + 1]
        self.layers = modelir.layers

        # An output parser is only useful for the last stage.
        # The outputs from the last stage need to be sent back to the inference
        # server. Therefore they need to be sent back as a list of tensors.
        # But if the output is a dictionary of tensors. This leads to comm
        # error. Also, in the inference, other values such as loss may not be
        # important. So, a way to manipulate the outputs is provided.
        self._output_parser: Union[Callable, None] = (
            modelir.output_parser if self.is_last else None
        )

        try:
            self._init_layers()
        except Exception as e:
            traceback.print_exc()
            raise e

        self._init_llm_config()

    def _init_llm_config(self):
        if not isinstance(self.modelir.mmd, Llama3ModelMetaData):
            return

        # further set up LLM causal LM parameters
        self.caches: dict[int, DynamicCache] = {}

        if self.is_full_model:
            self._run_llm = self._run_llm_full_model
            return

        if self.is_first:
            self._run_llm = self._run_llm_first_stage
        elif self.is_last:
            self._run_llm = self._run_llm_last_stage
        else:
            self._run_llm = self._run_llm_middle_stage

    def _run_llm_full_model(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        outputs = inputs

        while True:
            input_ids = outputs["input_ids"]
            attention_mask = outputs["attention_mask"]
            logger.debug(
                f"input_ids's size: {input_ids.size()} ",
                f"attention_mask's size: {attention_mask.size()}",
            )

            outputs = self.forward(
                **outputs,
                use_cache=True,
                past_key_values=cache,
            )

            outputs = self._output_parser(seqno, outputs, attention_mask)

            if "tokens" in outputs:
                # generating tokens is done
                break

        return outputs

    def _run_llm_first_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run the first stage of llm."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        logger.debug("run llm first stage")
        logger.debug(
            f"input_ids's size: {input_ids.size()} "
            + f"attention_mask's size: {attention_mask.size()}"
        )

        outputs = self.forward(**inputs, use_cache=True, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask

        return outputs

    def _run_llm_middle_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run a middle stage of llm."""
        logger.debug("run llm middle stage")
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to pass it to the next stage
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]

        outputs = self.forward(**inputs, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask

        return outputs

    def _run_llm_last_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run the last stage of llm."""
        logger.debug("run llm last stage")
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to use during output parsing
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]

        outputs = self.forward(**inputs, past_key_values=cache)

        outputs = self._output_parser(seqno, outputs, attention_mask)

        return outputs

    def _llm_generate(self, seqno: int, **inputs) -> tuple[dict[str, Tensor], int]:
        """Return generated intermediate results or all the tokens.

        Returns
        -------
        1st value: contains a dictionary of tensors
        2nd value: contains an index of layer that the results need to go back.
                   -1 means that the results goes back to the serving server.
        """
        # remove kv cache for batch that is already served to save memory
        # we do max_inflight+5 instead of max_inflight to be safe
        # TODO: An out-of-order llm request serving case can happen in infscale
        #       because it allows the arrival of other requests. Specifically,
        #       for the requests submitted late, if the llm produces shorter
        #       responses, the out-of-order serving is possible. The logic to
        #       remove kv caches of the served requests is built with assumption
        #       that the requests are served in order. This discrepancy sometimes
        #       creates an error at runtime. To fundamentally resolve the issue,
        #       we need an additional mechanism to determine which request is
        #       served. For now, to mitigate the issue, we lazily remove kv cache
        #       corresponding to a request whose sequence number is
        #       (seqno of the current request - (max_inflight + 5)).
        self.caches.pop(seqno - (self.max_inflight + 5), None)

        if seqno not in self.caches:
            self.caches[seqno] = DynamicCache()
        cache = self.caches[seqno]

        outputs = self._run_llm(seqno, cache, **inputs)
        # If DynamicCache is returned in outputs, it can't be forwarded
        # to other workers since it is not a tensor; so, we remove it
        # from outputs; this is a HACK; need to think about if there is
        # a better way to handle this
        outputs.pop("past_key_values", None)

        if self.is_last:
            # if tokens are in the outputs, token generation is done.
            # so, we can go back to the serving server
            # otherwise, outputs need to be fed into layer 0
            # due to auto-regressive nature of LLM's token generation
            next_layer = -1 if "tokens" in outputs else 0
        else:
            # if it's not the last layer or stage, we have to send outputs to
            # worker (or staage) that has the next layer
            next_layer = self.end + 1

        return outputs, next_layer

    def predict(self, seqno: int, **inputs) -> tuple[dict[str, Tensor], int]:
        """Coduct inference."""
        if isinstance(self.modelir.mmd, Llama3ModelMetaData):
            # do generation; needs multiple passes of the layers in a stateful manner
            # we have to maintain the state
            outputs, next_layer = self._llm_generate(seqno, **inputs)
        else:
            # run the layers once
            outputs = self.forward(**inputs)
            outputs = self._output_parser(outputs) if self._output_parser else outputs
            # other models like resnet don't have auto-regressive nature.
            # so, we can go back to the serving server
            next_layer = -1 if self.is_last else self.end + 1

        return outputs, next_layer

    def forward(self, **inputs) -> dict[str, Tensor]:
        """Run layers in the stage."""
        for layer in self.layers:
            inputs = layer(**inputs)

        return inputs

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
        layer: fx.GraphModule,
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
