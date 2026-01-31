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

"""ModelMetaData."""
from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Union

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForPreTraining,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from infscale import get_logger


AutoModelType = (
    AutoModelForPreTraining | AutoModelForCausalLM | AutoModelForImageClassification
)

logger = None


class ModelGroup(str, Enum):
    """Model class enum class."""

    UNKNOWN = "unknown"
    IMAGE = "image"
    LANG = "language"


class BaseModelMetaData:
    """Base class for model meta data implementation."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        global logger
        logger = get_logger()

        self.name: str = name
        self.model_group = ModelGroup.UNKNOWN
        self.config: PretrainedConfig = config

        self.model: AutoModelType = None
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
        self.split_points: List[str] = None
        self._trace_inputs: list[str] = None

    def _init_model(self, auto_model_type: AutoModelType):
        with init_empty_weights():
            model = auto_model_type.from_config(self.config)
            model.eval()  # disable training mode
            self.model_type = auto_model_type
            return model

    def load_model(self) -> AutoModelType:
        """Load a model onto cpu."""
        loaded_model = self.model_type.from_pretrained(self.name)

        return loaded_model

    @property
    def trace_inputs(self) -> list[str]:
        """Get input names to trace."""
        return self._trace_inputs

    @trace_inputs.setter
    def trace_inputs(self, names: list[str]) -> None:
        """Set input names to trace."""
        self._trace_inputs = names

    @abstractmethod
    def get_model(self) -> AutoModelType:
        """Abstract method to get model."""

    @abstractmethod
    def get_split_points(self) -> List[str]:
        """Abstract method to get split points."""

    @abstractmethod
    def get_output_parser(self) -> Union[Callable, None]:
        """Abstract method to return function to parse output."""

    @abstractmethod
    def get_predict_fn(self) -> Union[Callable, None]:
        """Abstract method to return function to predict."""


class Llama3ModelMetaData(BaseModelMetaData):
    """Llama3 model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, config=self.config)
        self.generated_tokens: dict[int, list[Tensor]] = {}
        self._init_gen_config()

    def _init_gen_config(self):
        self.bos_token_id = 128000
        if hasattr(self.config, "bos_token_id"):
            self.bos_token_id = self.config.bos_token_id
        logger.info(f"bos_token_id = {self.bos_token_id}")

        if hasattr(self.config, "do_sample"):
            self.do_sample = self.config.do_sample
        else:
            # Default to greedy decoding (matching vanilla transformers behavior)
            self.do_sample = False
        logger.info(f"do_sample = {self.do_sample}")

        self.eos_token_id = 128001
        if hasattr(self.config, "eos_token_id"):
            self.eos_token_id = self.config.eos_token_id
        logger.info(f"eos_token_id = {self.eos_token_id}")

        self.temperature = 0.6
        if hasattr(self.config, "temperature"):
            self.temperature = self.config.temperature
        logger.info(f"temperature = {self.temperature}")

        self.top_p = 0.9  # Changed from 1.0 to enable nucleus sampling
        if hasattr(self.config, "top_p"):
            self.top_p = self.config.top_p
        logger.info(f"top_p = {self.top_p}")

        self.max_new_tokens = 64

    @property
    def trace_inputs(self) -> list[str]:
        """Get input names to trace."""
        return self._trace_inputs

    @trace_inputs.setter
    def trace_inputs(self, names: list[str]) -> None:
        """Set input argument names to trace."""
        self._trace_inputs = names + [
            "use_cache",
            "past_key_values",
        ]
        logger.debug(f"llama3 trace inputs: {self._trace_inputs}")

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForCausalLM)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        # self.split_points.append("model.embed_tokens")
        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"model.layers.{i}")
        self.split_points.append("model.norm")
        self.split_points.append("lm_head")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""

        def inner(
            seqno: int, outputs: dict[str, Tensor], attention_mask: Tensor
        ) -> dict[str, Tensor]:
            # With left padding, all sequences are padded to the same length,
            # Extract logits for the last valid token position
            batch_size = outputs["logits"].size(0)
            seq_len = outputs["logits"].size(1)
            device = outputs["logits"].device
            
            # During generation with cache, logits shape is [batch, 1, vocab]
            # During initial pass, logits shape is [batch, full_seq_len, vocab]
            if seq_len == 1:
                # Generation step with cache: only one new token
                next_token_logits = outputs["logits"][:, 0, :]
            else:
                # Initial forward pass: need to find last valid token
                # For left padding: [0, 0, 1, 1, 1] → last valid at position 4 (index 4)
                # For right padding: [1, 1, 1, 0, 0] → last valid at position 2 (index 2)
                sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
                batch_indices = torch.arange(batch_size, device=device)
                next_token_logits = outputs["logits"][batch_indices, sequence_lengths, :]

            state = self.generated_tokens.get(seqno)
            if state is None:
                state = {
                    "tokens": [[] for _ in range(batch_size)],
                    "finished": [False] * batch_size,
                }
                self.generated_tokens[seqno] = state
                # CRITICAL: Clear KV cache when starting a new sample to prevent contamination
                if "past_key_values" in outputs:
                    outputs["past_key_values"] = None

            gen_tokens = state["tokens"]
            finished = state["finished"]

            if len(gen_tokens) != batch_size:
                raise ValueError(
                    "Mismatched batch size between cached tokens and new logits"
                )

            if self.do_sample:
                if self.temperature > 0.0:
                    # Apply temperature
                    next_token_logits = next_token_logits / self.temperature
                else:
                    self.do_sample = False
                    logger.info(f"temperature is 0.0, switching to greedy decoding")

            eos_token_id = self.eos_token_id

            for idx, is_finished in enumerate(finished):
                if is_finished:
                    next_token_logits[idx] = float("-inf")
                    next_token_logits[idx, eos_token_id] = 0.0

            if self.do_sample:
                if 0.0 < self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False

                    for batch_idx in range(batch_size):
                        if finished[batch_idx]:
                            continue
                        indices_to_remove = sorted_indices[batch_idx][
                            sorted_indices_to_remove[batch_idx]
                        ]
                        next_token_logits[batch_idx, indices_to_remove] = float(
                            "-inf"
                        )

                # Sample from the filtered distribution
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            for i in range(batch_size):
                if finished[i]:
                    next_token[i, 0] = eos_token_id
                    continue

                token_id = next_token[i, 0].item()
                gen_tokens[i].append(token_id)

                if token_id == eos_token_id or len(gen_tokens[i]) >= self.max_new_tokens:
                    finished[i] = True
                    if token_id != eos_token_id:
                        next_token[i, 0] = eos_token_id

            if all(finished):
                max_length = max(len(tokens) for tokens in gen_tokens)
                padded_tokens = []
                for tokens in gen_tokens:
                    if len(tokens) < max_length:
                        tokens = tokens + [eos_token_id] * (max_length - len(tokens))
                    padded_tokens.append(tokens)

                tensor = torch.tensor(padded_tokens, dtype=torch.int64, device=device)
                if tensor.size(0) == 1:
                    tensor = tensor[0]

                del self.generated_tokens[seqno]

                return {"tokens": tensor}

            # Update input_ids for the next iteration
            input_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        return inner

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""

        def inner(tensors: dict[str, Tensor]) -> list[str]:
            tokens = tensors["tokens"]
            tokens = tokens if len(tokens.shape) > 1 else [tokens]

            results = self.tokenizer.batch_decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            return results

        return inner


class Gpt2ModelMetaData(BaseModelMetaData):
    """Gpt2 model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    @property
    def trace_inputs(self) -> list[str]:
        """Get input names to trace."""
        return self._trace_inputs

    @trace_inputs.setter
    def trace_inputs(self, names: list[str]) -> None:
        """Set input argument names to trace."""
        self._trace_inputs = names

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForCausalLM)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"transformer.h.{i}")
        self.split_points.append("transformer.ln_f")
        # self.split_points.append("lm_head.weight")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""

        def inner(outputs):
            return outputs

        return inner

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.name, config=self.config
            )

        def inner(tensors: dict[str, Tensor]) -> list[str]:
            results = []
            tensors = tensors["logits"]
            for tensor in tensors:
                logits = tensor[-1, :]
                argmax = torch.argmax(logits)

                results.append(self.tokenizer.decode(argmax))
            return results

        return inner


class BertModelMetaData(BaseModelMetaData):
    """Bert model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForCausalLM)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"bert.encoder.layer.{i}")
        self.split_points.append("cls")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        raise NotImplementedError

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""
        raise NotImplementedError


class T5ModelMetaData(BaseModelMetaData):
    """T5 model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.LANG

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForPreTraining)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_layers):
            self.split_points.append(f"encoder.block.{i}")
        for i in range(self.config.num_decoder_layers):
            self.split_points.append(f"decoder.block.{i}")
        self.split_points.append("lm_head")

        logger.debug(f"#layers = {self.config.num_layers}")
        logger.debug(f"#decoder_layers = {self.config.num_decoder_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        raise NotImplementedError

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""
        raise NotImplementedError


class VitModelMetaData(BaseModelMetaData):
    """Vit model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.IMAGE

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForImageClassification)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Google's HuggingFace ViT model
        # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"vit.encoder.layer.{i}")
        self.split_points.append("vit.layernorm")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""
        raise NotImplementedError

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""
        raise NotImplementedError


class ResnetModelMetaData(BaseModelMetaData):
    """Resnet model meta data class."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        super().__init__(name, config)

        self.model_group = ModelGroup.IMAGE

    def get_model(self) -> AutoModelType:
        """Get model."""
        if self.model:
            return self.model

        self.model = self._init_model(AutoModelForImageClassification)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Microsoft's HuggingFace ResNet model
        # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i, depth in enumerate(self.config.depths):
            for j in range(depth):
                self.split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")

        self.split_points.append("resnet.pooler")

        logger.debug(f"#depths = {sum(self.config.depths)}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""

        def inner(outputs):
            return outputs

        return inner

    def get_predict_fn(self) -> Union[Callable, None]:
        """Return function to predict."""

        def inner(tensors: dict[str, Tensor]) -> list[str]:
            results = []

            tensors = tensors["logits"]
            for tensor in tensors:
                predicted_label = tensor.argmax(-1).item()
                results.append(self.config.id2label[predicted_label])
            return results

        return inner
