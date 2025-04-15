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
        seq_length: int = -1,
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
        self.profile_count = 0
        self.max_profile_count = max_profile_count
        self.profile_data_saved = False
        # For non-LLM or overall timing
        self.layer_times = {}
        # For LLM specific timing
        self.layer_times_prefill = {}
        self.layer_times_decode = {}
        self.seqno_counters = {}
        self.sequences_being_profiled = set()
        self.profiled_prefill_count = 0

        self.model_name = model_name.split("/")[-1]
        self.batch_size = batch_size
        self.seq_length = seq_length if seq_length != -1 else None

        # If profiling is enabled, create events for each layer
        if self.profile:
            assert torch.cuda.is_available(), "Profiling requires CUDA"
            self.start_events = {}
            self.end_events = {}

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

        outputs, measured_times = self.forward(**inputs, use_cache=True, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask

        return outputs, measured_times

    def _run_llm_middle_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run a middle stage of llm."""
        logger.debug("run llm middle stage")
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to pass it to the next stage
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]

        outputs, measured_times = self.forward(**inputs, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask

        return outputs, measured_times

    def _run_llm_last_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run the last stage of llm."""
        logger.debug("run llm last stage")
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to use during output parsing
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]

        outputs, measured_times = self.forward(**inputs, past_key_values=cache)

        outputs = self._output_parser(seqno, outputs, attention_mask)

        return outputs, measured_times

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

        outputs, measured_times = self._run_llm(seqno, cache, **inputs)

        # Handle profiling for LLMs
        if self.profile and measured_times is not None:
            
            # Determine if this sequence should be profiled by this stage
            should_profile_this_seqno = False
            if seqno in self.sequences_being_profiled:
                should_profile_this_seqno = True
            elif len(self.sequences_being_profiled) < self.max_profile_count:
                # Start profiling this new sequence if we haven't reached the limit
                should_profile_this_seqno = True
                self.sequences_being_profiled.add(seqno)

            if should_profile_this_seqno:
                self.seqno_counters[seqno] = self.seqno_counters.get(seqno, 0) + 1
                is_prefill = self.seqno_counters[seqno] == 1

                # Record timings
                for layer_idx, time in measured_times.items():
                    if is_prefill:
                        self.layer_times_prefill.setdefault(layer_idx, []).append(time)
                    else:
                        self.layer_times_decode.setdefault(layer_idx, []).append(time)

                # If this was a prefill step, increment count and check save condition
                if is_prefill:
                    self.profiled_prefill_count += 1
                    # Save profiling data if we've recorded enough prefill steps
                    if self.profiled_prefill_count >= self.max_profile_count and self.max_profile_count > 0 and not self.profile_data_saved:
                        self._save_profile_data()
                        # self.profile_data_saved is set within _save_profile_data()

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
        is_llm = isinstance(self.modelir.mmd, Llama3ModelMetaData)

        if not self.profile:
            return
            
        layer_indices = set(self.start_events.keys()) # Get all layer indices in this stage

        profile_data = []
        profile_data_prefill = []
        profile_data_decode = []


        for layer_idx in sorted(list(layer_indices)):

            if is_llm:
                layer_data_prefill = {
                    "layer_num": layer_idx,
                    "layer_name": self.layers[layer_idx - self.start].__class__.__name__,
                    "type": "prefill",
                }
                layer_data_decode = {
                    "layer_num": layer_idx,
                    "layer_name": self.layers[layer_idx - self.start].__class__.__name__,
                    "type": "decode",
                }

                times_prefill = self.layer_times_prefill.get(layer_idx, [])
                times_decode = self.layer_times_decode.get(layer_idx, [])

                def calculate_stats(times):
                    assert len(times) > 20, f"max_profile_count is too small, should be more than 20 as we remove the max and min 10 values"
                    times.sort()
                    # TODO: Consider making the number of points to remove configurable
                    times_trimmed = times[10:-10] # remove the max and min ten values
                    if not times_trimmed:
                        return None, None, None, None # Handle case where trimming removed everything
                    avg_time = sum(times_trimmed) / len(times_trimmed)
                    min_time = min(times_trimmed)
                    max_time = max(times_trimmed)
                    std_dev = statistics.stdev(times_trimmed) if len(times_trimmed) > 1 else 0
                    return avg_time, min_time, max_time, std_dev

                avg_prefill, min_prefill, max_prefill, std_prefill = calculate_stats(times_prefill)
                avg_decode, min_decode, max_decode, std_decode = calculate_stats(times_decode)

                if avg_prefill is not None:
                    layer_data_prefill.update({
                        "forward_latency_ms": avg_prefill,
                        "min_prefill": min_prefill,
                        "max_prefill": max_prefill,
                        "std_prefill": std_prefill,
                        "count_prefill": len(times_prefill),
                    })
                if avg_decode is not None:
                    layer_data_decode.update({
                        "forward_latency_ms": avg_decode,
                        "min_decode": min_decode,
                        "max_decode": max_decode,
                        "std_decode": std_decode,
                        "count_decode": len(times_decode),
                    })
                # Only add layer if we have either prefill or decode data
                if avg_prefill is not None or avg_decode is not None:
                   profile_data_prefill.append(layer_data_prefill)
                   profile_data_decode.append(layer_data_decode)

            else: # Not LLM
                layer_data = {
                    "layer_num": layer_idx,
                    "layer_name": self.layers[layer_idx - self.start].__class__.__name__,
                }

                times = self.layer_times.get(layer_idx, [])
                # Calculate average time where the max and min ten values are removed
                times.sort()
                assert len(times) > 20, f"max_profile_count is too small, should be more than 20 as we remove the max and min 10 values"
                times_trimmed = times[10:-10] # remove the max and min ten values
                if not times_trimmed:
                    continue
                avg_time = sum(times_trimmed) / len(times_trimmed)
                min_time = min(times_trimmed)
                max_time = max(times_trimmed)
                std_dev = statistics.stdev(times_trimmed) if len(times_trimmed) > 1 else 0

                layer_data.update({
                    "forward_latency_ms": avg_time,
                    "min": min_time,
                    "max": max_time,
                    "std": std_dev,
                    "count": len(times),
                })
                profile_data.append(layer_data)

        # Save to file
        
        if is_llm:
            profile_dir_decode = Path("profile_data") / "infscale" / f"{self.model_name}_decode"
            profile_dir_decode.mkdir(exist_ok=True, parents=True)
            profile_dir_prefill = Path("profile_data") / "infscale" / f"{self.model_name}_prefill"
            profile_dir_prefill.mkdir(exist_ok=True, parents=True)
        else:
            profile_dir = Path("profile_data") / "infscale" / self.model_name
            profile_dir.mkdir(exist_ok=True, parents=True)
        
        output_filename = f"batch_size_{self.batch_size}_stage_{self.id}.json"

        if is_llm and self.seq_length is not None:
            output_filename = f"batch_size_{self.batch_size}_seq_length_{self.seq_length}_stage_{self.id}.json"


        if is_llm:
            save_path_decode = profile_dir_decode / output_filename
            save_path_prefill = profile_dir_prefill / output_filename
            with open(save_path_decode, 'w') as f:
                json.dump(profile_data_decode, f, indent=2)
            with open(save_path_prefill, 'w') as f:
                json.dump(profile_data_prefill, f, indent=2)

            print(f"Saved profiling data for stage {self.id} to {profile_dir_decode} and {profile_dir_prefill}")
        else:
            save_path = profile_dir / output_filename
            with open(save_path, 'w') as f:
                json.dump(profile_data, f, indent=2)

            print(f"Saved profiling data for stage {self.id} to {save_path}")

        self.profile_data_saved = True # Mark as saved

    def predict(self, seqno: int, **inputs) -> tuple[dict[str, Tensor], int]:
        """Coduct inference."""
        is_llm = isinstance(self.modelir.mmd, Llama3ModelMetaData)

        if is_llm:
            # do generation; needs multiple passes of the layers in a stateful manner
            # we have to maintain the state
            outputs, next_layer = self._llm_generate(seqno, **inputs)
        else:
            # run the layers once
            outputs, measured_times = self.forward(**inputs)

            # Handle profiling for non-LLM models
            if self.profile and torch.cuda.is_available() and measured_times is not None:
                self.profile_count += 1 # Increment non-LLM profile counter
                for layer_idx, time in measured_times.items():
                    self.layer_times.setdefault(layer_idx, []).append(time)

                # Save profiling data if we've reached max_count for non-LLM runs
                if self.profile_count >= self.max_profile_count and self.max_profile_count > 0 and not self.profile_data_saved:
                    self._save_profile_data()
                    # self.profile_data_saved is set within _save_profile_data

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
            return inputs, None
        
        # Forward pass with profiling
        measured_times = {}
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
            measured_times[layer_idx] = elapsed_time
        
        return inputs, measured_times
    
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
