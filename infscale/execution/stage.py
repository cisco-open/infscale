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

import statistics
import traceback
import json
import os
from typing import TYPE_CHECKING, Callable, Union, Dict, List
from pathlib import Path

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
        profile: bool = False,
        max_profile_count: int = 100,
        model_name: str = "",
        batch_size: int = 1,
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
        
        self.profile = profile
        self.layer_profile_data = []
        self.profile_count = 0
        self.max_profile_count = max_profile_count
        self.profile_data_saved = False

        self.model_name = model_name.split("/")[-1]
        self.batch_size = batch_size
        
        # If profiling is enabled, create events for each layer
        if self.profile and torch.cuda.is_available():
            self.start_events = {}
            self.end_events = {}
            self.layer_times = {}

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
        
        # Initialize profiling events if enabled
        if self.profile and torch.cuda.is_available():
            for i, layer in enumerate(self.layers):
                layer_idx = i + self.start
                self.start_events[layer_idx] = torch.cuda.Event(enable_timing=True)
                self.end_events[layer_idx] = torch.cuda.Event(enable_timing=True)
                self.layer_times[layer_idx] = []

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

    def _save_profile_data(self):
        """Save profiling data to JSON file."""
        if not self.profile or not self.layer_times:
            return
            
        profile_dir = Path("profile_data") / self.model_name / f"batch_size_{self.batch_size}"
        profile_dir.mkdir(exist_ok=True, parents=True)
        
        profile_data = []
        
        for layer_idx, times in self.layer_times.items():
            if not times:
                continue
                
            # Calculate average time where the max and min two values are removed
            times.sort()
            assert len(times) > 20, f"max_profile_count is too small, should be more than 20 as we remove the max and min 10 values"
            times = times[10:-10] # remove the max and min five values
            avg_time = sum(times) / len(times)
            
            layer_data = {
                "layer_num": layer_idx,
                "layer_name": self.layers[layer_idx - self.start].__class__.__name__,
                "forward_latency_ms": avg_time,
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times),
            }
            profile_data.append(layer_data)
        
        # Save to file
        with open(profile_dir / f"profile_stage_{self.id}.json", 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"Saved profiling data for stage {self.id} to profile_data/infscale_profile/{self.model_name}/batch_size_{self.batch_size}/profile_stage_{self.id}.json")

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
        if not self.profile or not torch.cuda.is_available():
            # Standard forward pass without profiling
            for layer in self.layers:
                inputs = layer(**inputs)
            return inputs
        
        # Forward pass with profiling
        for i, layer in enumerate(self.layers):
            layer_idx = i + self.start
            self.start_events[layer_idx].record()
            
            inputs = layer(**inputs)
            
            self.end_events[layer_idx].record()
        
        # Synchronize CUDA and calculate times
        torch.cuda.synchronize()
        
        for i, _ in enumerate(self.layers):
            layer_idx = i + self.start
            elapsed_time = self.start_events[layer_idx].elapsed_time(self.end_events[layer_idx])
            self.layer_times[layer_idx].append(elapsed_time)
        
        # Increment profile count
        self.profile_count += 1
        
        # Save profiling data if we've reached max_count
        if self.profile_count >= self.max_profile_count and self.max_profile_count > 0 and not self.profile_data_saved:
            self._save_profile_data()
            self.profile_data_saved = True

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
