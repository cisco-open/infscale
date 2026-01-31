# Copyright 2026 Cisco Systems, Inc. and its affiliates
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

"""
Manual model sharding without torch.fx tracing.

This approach directly extracts layers from the model hierarchy,
avoiding torch.fx limitations with control flow and dynamic behavior.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from infscale import get_logger
from infscale.module.model_metadata import BaseModelMetaData, Llama3ModelMetaData


logger = get_logger(__name__)


class ManualSharder:
    """Manual model sharding without symbolic tracing."""

    @classmethod
    def shard(cls, mmd: BaseModelMetaData) -> List[nn.Module]:
        """
        Return a list of layer objects that can be sharded.

        Returns a flat list of individual "layers" (operations) that the
        Stage class will slice based on start/end indices from config.
        """
        split_points = mmd.get_split_points()

        assert (
            split_points
        ), f"Empty split points. Check model type {mmd.config.model_type} is supported."

        model = mmd.get_model()

        # Dispatch to model-specific sharding
        if isinstance(mmd, Llama3ModelMetaData):
            return cls._shard_llama(model, split_points)
        else:
            raise NotImplementedError(
                f"Manual sharding not implemented for {type(mmd).__name__}. "
                f"Model type: {mmd.config.model_type}"
            )

    @classmethod
    def _shard_llama(
        cls, model: PreTrainedModel, split_points: List[str]
    ) -> List[nn.Module]:
        """
        Extract LLaMA decoder layers as individual wrapped modules.

        Returns 32 wrapped decoder layers (simpler than torch.fx's 35 operations).
        Each wrapper handles embeddings (first layer), decoder layer, and norm+lm_head (last layer).

        Args:
            model: LlamaForCausalLM model
            split_points: Not used for manual sharding, but kept for API compatibility

        Returns:
            List of 32 wrapped decoder layer modules
        """
        logger.info(f"Manual sharding LLaMA model")

        # Get model components
        base_model = model.model  # LlamaModel
        all_layers = list(base_model.layers)  # ModuleList of LlamaDecoderLayer
        num_layers = len(all_layers)

        logger.info(f"Total decoder layers in model: {num_layers}")

        # Create list of wrapped layers (32 total)
        wrapped_layers = []

        for idx, layer in enumerate(all_layers):
            wrapped = LlamaLayerWrapper(
                layer=layer,
                layer_idx=idx,
                operation=f"layer_{idx}",
                embed_tokens=base_model.embed_tokens if idx == 0 else None,
                rotary_emb=base_model.rotary_emb,  # ALL layers need rotary_emb!
                norm=base_model.norm if idx == num_layers - 1 else None,
                lm_head=model.lm_head if idx == num_layers - 1 else None,
                config=model.config,
                llama_model=(
                    base_model if idx == 0 else None
                ),  # Only first layer needs model reference
            )
            wrapped_layers.append(wrapped)

        logger.info(f"Created {len(wrapped_layers)} wrapped layer modules")

        return wrapped_layers


class LlamaLayerWrapper(nn.Module):
    """Wrapper for a single LLaMA decoder layer with special operations."""

    def __init__(
        self,
        layer: Optional[nn.Module],
        layer_idx: int,
        operation: str,
        embed_tokens: Optional[nn.Module] = None,
        rotary_emb: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
        config=None,
        llama_model: Optional[
            nn.Module
        ] = None,  # Reference to full LlamaModel for _update_causal_mask
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.operation = operation
        self.config = config

        # Store decoder layer
        self.layer = layer

        # Store special components
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.norm = norm
        self.lm_head = lm_head

        # Store reference to LlamaModel without registering it as a submodule
        # Use object.__setattr__ to bypass PyTorch's module registration
        object.__setattr__(self, "_llama_model_ref", llama_model)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass for a wrapped decoder layer.

        First layer (idx 0): embeddings + decoder layer
        Middle layers: decoder layer only
        Last layer (idx 31): decoder layer + norm + lm_head
        """

        # Handle torch.fx output format: {0: hidden_states}
        # Note: When checking kwargs, 0 might be passed as integer key
        if hidden_states is None:
            if 0 in kwargs:
                hidden_states = kwargs.pop(0)
            elif "hidden_states" in kwargs:
                hidden_states = kwargs.pop("hidden_states")

        # Extract position/cache info from kwargs if not provided as direct args
        if position_ids is None and "position_ids" in kwargs:
            position_ids = kwargs.get("position_ids")
        if cache_position is None and "cache_position" in kwargs:
            cache_position = kwargs.get("cache_position")
        if past_key_values is None and "past_key_values" in kwargs:
            past_key_values = kwargs.get("past_key_values")

        # First layer: Handle embeddings and causal mask
        if self.layer_idx == 0 and self.embed_tokens is not None:
            if input_ids is None:
                raise ValueError("First layer requires input_ids")

            # Embed tokens
            hidden_states = self.embed_tokens(input_ids)

            # Compute position IDs and cache_position if not provided
            if cache_position is None:
                # Initial pass: create cache_position for the full sequence
                past_seen = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                seq_length = input_ids.shape[1]
                cache_position = torch.arange(
                    past_seen,
                    past_seen + seq_length,
                    dtype=torch.long,
                    device=input_ids.device,
                )

            if position_ids is None:
                # Compute position_ids based on attention_mask to support left padding
                if attention_mask is not None and past_seen == 0:
                    # Initial forward pass with left padding:
                    # attention_mask: [0, 0, 1, 1, 1] for left-padded sequence
                    # cumsum: [0, 0, 1, 2, 3]
                    # minus 1: [-1, -1, 0, 1, 2]
                    # masked_fill with 0: [0, 0, 0, 1, 2] → valid tokens get positions [0, 1, 2]
                    # Compute position_ids for left padding
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                elif attention_mask is not None and past_seen > 0:
                    # Generation with cache: compute per-sample position based on real tokens
                    # HuggingFace debug shows: position_ids = attention_mask.sum(dim=1) - 1
                    # Example from debug:
                    #   attention_mask sum=7 (7 tokens at positions 0-6) → position_ids=[6]
                    #   attention_mask sum=3 (3 real tokens) → position_ids=[2]
                    # This gives the position of the CURRENT token (0-indexed)
                    sequence_lengths = (
                        attention_mask.sum(dim=1, keepdim=True) - 1
                    )  # [batch_size, 1]
                    position_ids = sequence_lengths.long()
                else:
                    # Right padding or no attention mask: use cache_position
                    position_ids = cache_position.unsqueeze(0)

            # CRITICAL: Compute causal mask from attention_mask (like LlamaModel does)
            # Only compute during initial pass (past_seen == 0)
            # During generation, pass None - decoder layers handle causal attention internally
            llama_model_ref = object.__getattribute__(self, "_llama_model_ref")
            if (
                llama_model_ref is not None
                and attention_mask is not None
                and past_seen == 0
            ):
                causal_mask = llama_model_ref._update_causal_mask(
                    attention_mask,
                    hidden_states,
                    cache_position,
                    past_key_values,
                    output_attentions,
                )

                # If _update_causal_mask returns None but we have attention_mask (for left padding),
                # we need to manually create a 4D causal mask
                if causal_mask is None and attention_mask is not None:
                    from transformers.modeling_attn_mask_utils import (
                        AttentionMaskConverter,
                    )

                    mask_converter = AttentionMaskConverter(
                        is_causal=True, sliding_window=None
                    )
                    seq_length = hidden_states.shape[1]
                    causal_mask = mask_converter.to_4d(
                        attention_mask,  # 2D attention mask
                        query_length=seq_length,
                        dtype=hidden_states.dtype,
                        key_value_length=seq_length,
                    )
            else:
                causal_mask = None
        else:
            if hidden_states is None:
                raise ValueError(f"Layer {self.layer_idx} requires hidden_states")
            # Middle/last layers: receive causal_mask and attention_mask from previous layer
            causal_mask = kwargs.get("causal_mask", None)
            # CRITICAL: Get attention_mask from kwargs for subsequent layers
            if attention_mask is None:
                attention_mask = kwargs.get("attention_mask", None)

        # Compute position embeddings (RoPE) if not provided
        # In full-model mode: computed once in layer 0, then reused
        # In pipeline mode: each stage computes it for its first layer (not passed between stages)
        position_embeddings = kwargs.get("position_embeddings", None)
        if (
            position_embeddings is None
            and self.rotary_emb is not None
            and position_ids is not None
        ):
            # Compute position embeddings when not available
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through decoder layer
        # Pass causal_mask (either 4D mask or None)
        # When None, the attention layer will handle causal masking internally
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=causal_mask,  # Pass 4D causal mask or None (NOT 2D attention_mask)
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

        # Last layer: Apply norm and lm_head
        if self.norm is not None and self.lm_head is not None:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            return {
                "logits": logits,
            }

        # Regular output: return hidden states
        # Use "hidden_states" as key (string, not integer 0)
        outputs = {
            "hidden_states": hidden_states,
        }

        # Pass essential info to next layers
        if position_ids is not None:
            outputs["position_ids"] = position_ids
        if cache_position is not None:
            outputs["cache_position"] = cache_position
        if causal_mask is not None:
            outputs["causal_mask"] = (
                causal_mask  # Pass computed causal mask to next layers
            )
        # CRITICAL: Pass attention_mask forward! Each layer needs it for attention computation
        if attention_mask is not None:
            outputs["attention_mask"] = attention_mask
        # CRITICAL: Pass position_embeddings forward! All layers need the same embeddings
        if position_embeddings is not None:
            outputs["position_embeddings"] = position_embeddings
        # CRITICAL: Pass past_key_values forward! All layers need the same cache object (updated in-place)
        if past_key_values is not None:
            outputs["past_key_values"] = past_key_values

        return outputs
