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

import time
import traceback
from typing import TYPE_CHECKING, Callable, Union

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from torch.nn import Parameter
from transformers import DynamicCache

from infscale import get_logger
from infscale.common.constants import EVICT_SEQNO_KEY
from infscale.module.model_metadata import Llama3ModelMetaData
from infscale.module.modelir import ModelIR


if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


logger = None

# Maximum number of evictions between torch.cuda.empty_cache() calls.
# Caps the interval so that large max_inflight values do not delay
# cache reclamation and risk OOM.
EMPTY_CACHE_INTERVAL = 4


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
        # Cache structure: seqno -> (DynamicCache, last_access_timestamp)
        self.caches: dict[int, tuple[DynamicCache, float]] = {}

        # EMA tracking for adaptive timeout
        self.ema_inter_arrival = None  # EMA of inter-arrival times
        self.ema_alpha = 0.5  # Smoothing factor (increased for faster adaptation to load changes)
        self.last_request_time = None  # Track last request timestamp

        # Adaptive timeout configuration
        self.timeout_multiplier = 10  # Safety factor (increased to handle pipeline round-trip)
        self.adaptive_timeout = 10.0  # Initial default
        self.min_timeout = 0.5  # Lower bound
        self.max_timeout = 30.0  # Upper bound

        # Cleanup configuration
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 0.5  # Check every 0.5s

        # Track evictions to amortize torch.cuda.empty_cache() calls
        self._evict_count = 0


        if self.is_full_model:
            self._run_llm = self._run_llm_full_model
            return

        if self.is_first:
            self._run_llm = self._run_llm_first_stage
        elif self.is_last:
            self._run_llm = self._run_llm_last_stage
        else:
            self._run_llm = self._run_llm_middle_stage

    def _update_adaptive_timeout(self) -> None:
        """Update adaptive timeout using EMA of inter-arrival times."""
        current_time = time.time()

        # Calculate inter-arrival time
        if self.last_request_time is not None:
            inter_arrival = current_time - self.last_request_time

            # Update EMA
            if self.ema_inter_arrival is None:
                self.ema_inter_arrival = inter_arrival
            else:
                self.ema_inter_arrival = (
                    self.ema_alpha * inter_arrival
                    + (1 - self.ema_alpha) * self.ema_inter_arrival
                )

            # Compute timeout: EMA × 2 × max_inflight
            timeout = (
                self.ema_inter_arrival * self.timeout_multiplier * self.max_inflight
            )

            # Apply bounds
            self.adaptive_timeout = max(
                self.min_timeout, min(self.max_timeout, timeout)
            )

        self.last_request_time = current_time

    def _cleanup_idle_caches(self) -> None:
        """Remove caches for completed requests based on idle time."""
        current_time = time.time()

        # Rate limit cleanup checks
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return

        self.last_cleanup_time = current_time

        # Find and evict idle caches
        to_evict = [
            seqno
            for seqno, (cache, last_access) in self.caches.items()
            if current_time - last_access > self.adaptive_timeout
        ]

        # Evict
        for seqno in to_evict:
            del self.caches[seqno]

        if to_evict:
            self._evict_count += 1
            if (
                self._evict_count % EMPTY_CACHE_INTERVAL == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

    def _run_llm_full_model(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        outputs = inputs

        while True:
            input_ids = outputs["input_ids"]
            attention_mask = outputs["attention_mask"]

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

        outputs = self.forward(**inputs, use_cache=True, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask
        
        # Remove non-serializable objects before sending to next stage
        outputs.pop("past_key_values", None)  # DynamicCache can't be serialized
        outputs.pop("position_embeddings", None)  # Tuple can't be serialized
        outputs.pop("use_cache", None)  # Boolean can't be serialized

        return outputs

    def _run_llm_middle_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run a middle stage of llm."""
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to pass it to the next stage
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]
        
        outputs = self.forward(**inputs, use_cache=True, past_key_values=cache)

        # add attention mask to outputs to pass it to next stage
        outputs["attention_mask"] = attention_mask
        
        # Remove non-serializable objects before sending to next stage
        outputs.pop("past_key_values", None)  # DynamicCache can't be serialized
        outputs.pop("position_embeddings", None)  # Tuple can't be serialized
        outputs.pop("use_cache", None)  # Boolean can't be serialized

        return outputs

    def _run_llm_last_stage(
        self, seqno: int, cache: DynamicCache, **inputs
    ) -> dict[str, Tensor]:
        """Run the last stage of llm."""
        # attention mask passed from the upstream stage shouldn't be used
        # during inference. we save it to use during output parsing
        attention_mask = inputs["attention_mask"]
        del inputs["attention_mask"]

        outputs = self.forward(**inputs, use_cache=True, past_key_values=cache)
        
        # Remove non-serializable objects before parsing
        outputs.pop("past_key_values", None)  # DynamicCache can't be serialized
        outputs.pop("position_embeddings", None)  # Tuple can't be serialized
        outputs.pop("use_cache", None)  # Boolean can't be serialized

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
        next_layer = -1 if self.is_last else self.end + 1

        # Explicit evict (LLM-only): evict KV cache and forward sentinel
        if EVICT_SEQNO_KEY in inputs:
            if seqno in self.caches:
                del self.caches[seqno]
                self._evict_count += 1
                if (
                    self._evict_count % EMPTY_CACHE_INTERVAL == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
            return ({EVICT_SEQNO_KEY: seqno}, next_layer)

        # Update adaptive timeout based on inter-arrival pattern
        self._update_adaptive_timeout()

        # Periodic cleanup of idle caches
        self._cleanup_idle_caches()

        # Get or create cache with timestamp
        current_time = time.time()
        if seqno not in self.caches:
            cache = DynamicCache()
            self.caches[seqno] = (cache, current_time)
        else:
            cache, _ = self.caches[seqno]
            self.caches[seqno] = (cache, current_time)  # Update timestamp

        outputs = self._run_llm(seqno, cache, **inputs)
        # If DynamicCache is returned in outputs, it can't be forwarded
        # to other workers since it is not a tensor; so, we remove it
        # from outputs; this is a HACK; need to think about if there is
        # a better way to handle this
        outputs.pop("past_key_values", None)

        return outputs, next_layer

    def predict(self, seqno: int, **inputs) -> tuple[dict[str, Tensor], int]:
        """Conduct inference."""
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
        model.eval()  # CRITICAL: Set to eval mode to disable dropout and other training-time operations

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

        for idx, layer in enumerate(self.layers):
            self._init_tensors(layer, named_parameters, named_buffers)

        del named_parameters
        del named_buffers
        del model

    def _init_tensors(
        self,
        layer: torch.nn.Module,
        named_parameters: dict[str, Parameter],
        named_buffers: dict[str, Tensor],
    ):
        """Initialize meta tensors and move them to a device."""
        # Check if this is a manual sharder wrapper (has layer_idx attribute)
        is_manual_wrapper = hasattr(layer, 'layer_idx')
        
        param_count = 0
        for name, _ in layer.named_parameters():
            # Map wrapper parameter names to model parameter names
            if is_manual_wrapper:
                model_param_name = self._map_wrapper_param_name(name, layer.layer_idx)
            else:
                model_param_name = name
            
            assert model_param_name in named_parameters, f"parameter {model_param_name} not found (wrapper name: {name})"

            set_module_tensor_to_device(
                layer,
                name,
                self.device,
                named_parameters[model_param_name].data,
            )
            param_count += 1
        
        for name, _ in layer.named_buffers():
            # Map wrapper buffer names to model buffer names
            if is_manual_wrapper:
                model_buffer_name = self._map_wrapper_param_name(name, layer.layer_idx)
            else:
                model_buffer_name = name
            
            assert model_buffer_name in named_buffers, f"buffer {model_buffer_name} not found (wrapper name: {name})"

            set_module_tensor_to_device(
                layer,
                name,
                self.device,
                named_buffers[model_buffer_name].data,
            )
    
    def _map_wrapper_param_name(self, wrapper_name: str, layer_idx: int) -> str:
        """
        Map manual sharder wrapper parameter name to model parameter name.
        
        Wrapper names: layer.self_attn.q_proj.weight, embed_tokens.weight, etc.
        Model names: model.layers.{idx}.self_attn.q_proj.weight, model.embed_tokens.weight, etc.
        """
        if wrapper_name.startswith("layer."):
            # Decoder layer parameter: layer.* -> model.layers.{idx}.*
            param_name = wrapper_name[6:]  # Remove "layer." prefix
            return f"model.layers.{layer_idx}.{param_name}"
        elif wrapper_name.startswith("embed_tokens"):
            # Embedding: embed_tokens -> model.embed_tokens or embed_tokens.weight -> model.embed_tokens.weight
            return f"model.{wrapper_name}"
        elif wrapper_name.startswith("rotary_emb"):
            # RoPE: rotary_emb.* -> model.rotary_emb.*
            return f"model.{wrapper_name}"
        elif wrapper_name.startswith("norm"):
            # Norm: norm -> model.norm or norm.weight -> model.norm.weight
            return f"model.{wrapper_name}"
        elif wrapper_name.startswith("lm_head"):
            # LM head: lm_head.* stays the same
            return wrapper_name
        else:
            # Unknown, return as-is
            return wrapper_name
