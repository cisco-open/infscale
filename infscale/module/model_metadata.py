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
from infscale import get_logger
from torch import Tensor
from transformers import (AutoModelForCausalLM,
                          AutoModelForImageClassification,
                          AutoModelForPreTraining, AutoTokenizer,
                          PretrainedConfig, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

AutoModelType = (
    AutoModelForPreTraining | AutoModelForCausalLM | AutoModelForImageClassification
)

logger = get_logger()


class ModelGroup(str, Enum):
    """Model class enum class."""

    UNKNOWN = "unknown"
    IMAGE = "image"
    LANG = "language"


class BaseModelMetaData:
    """Base class for model meta data implementation."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
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
        # let's keep sampling true
        self.do_sample = True
        logger.info(f"do_sample = {self.do_sample}")

        self.eos_token_id = 128001
        if hasattr(self.config, "eos_token_id"):
            self.eos_token_id = self.config.eos_token_id
        logger.info(f"eos_token_id = {self.eos_token_id}")

        self.temperature = 0.6
        if hasattr(self.config, "temperature"):
            self.temperature = self.config.temperature
        logger.info(f"temperature = {self.temperature}")

        self.top_p = 1.0
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
        logger.info(f"llama3 trace inputs: {self._trace_inputs}")

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

        logger.info(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points

    def get_output_parser(self) -> Union[Callable, None]:
        """Return function to parse output."""

        def inner(
            seqno: int, outputs: dict[str, Tensor], attention_mask: Tensor
        ) -> dict[str, Tensor]:
            next_token_logits = outputs["logits"][:, -1, :]
            device = next_token_logits.device

            if self.do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / self.temperature

                # Apply top_p (nucleus) sampling
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
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float("-inf")

                # Sample from the filtered distribution
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if seqno not in self.generated_tokens:
                self.generated_tokens[seqno] = []
            gen_token_list = self.generated_tokens[seqno]

            gen_token_list.append(next_token.item())
            # TODO: remove this print statement or convert it to a log statment
            #       this is for temporary use to help understanding of the
            #       generation process
            print(f">>> generated {len(gen_token_list)}th token for batch {seqno}")

            # Check for EOS token or if max number of tokens are generated
            if (
                self.max_new_tokens == len(gen_token_list)
                or next_token.item() == self.eos_token_id
            ):
                tensor = torch.tensor(gen_token_list, dtype=torch.int64, device=device)
                gen_token_list = []
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
            results = []
            tokens = tensors["tokens"]

            results.append(self.tokenizer.decode(tokens))

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
