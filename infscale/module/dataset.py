"""
Copyright (c) 2023 SymbioticLab, The University of Michigan

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This file was modified from
# https://github.com/SymbioticLab/Oobleck/blob/3b7a0c2f19bff0991e623ffbeb8a5b365853bf3a/oobleck/execution/dataset.py

import math
from typing import Optional, Tuple, Type

import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizerBase

from infscale import get_logger
from infscale.module.model_metadata import BaseModelMetaData, ModelGroup


logger = None


class HuggingFaceDataset:
    """
    Load datasets from Hugging Face Hub (https://huggingface.co/datasets)
    and do preprocessing.
    """

    def __init__(
        self,
        mmd: BaseModelMetaData,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        split: Optional[str] = "test",
        max_seq_length: Optional[int] = None,
    ):
        """Initialize the class."""
        global logger
        logger = get_logger()

        if mmd.model_group == ModelGroup.LANG:
            self.tokenizer, self.dataset = HuggingFaceDataset.create_language_dataset(
                mmd.name,
                dataset_path,
                dataset_name,
                split,
                max_seq_length,
            )

            def collate_fn(examples):
                input_ids = []
                attention_mask = []
                for ex in examples:
                    ids = torch.tensor(ex["input_ids"], dtype=torch.int64)
                    input_ids.append(ids)
                    mask = torch.tensor(ex["attention_mask"], dtype=torch.int64)
                    attention_mask.append(mask)

                return {
                    "input_ids": torch.stack(input_ids),
                    "attention_mask": torch.stack(attention_mask),
                }

            self.data_collator = collate_fn

        elif mmd.model_group == ModelGroup.IMAGE:
            self.tokenizer, self.dataset = HuggingFaceDataset.create_image_dataset(
                mmd.name,
                dataset_path,
                dataset_name,
                split,
            )

            def collate_fn(examples):
                pixel_values = torch.stack(
                    [example["pixel_values"] for example in examples]
                )
                return {"pixel_values": pixel_values}

            self.data_collator = collate_fn

        else:
            self.dataset = None

        assert (
            self.dataset
        ), f"Dataset uninitialized due to unsupported model {mmd.name}."

        sample = self.data_collator([next(iter(self.dataset))])
        trace_inputs = list(sample.keys())
        mmd.trace_inputs = trace_inputs

        self.model_group = mmd.model_group

    def configure(
        self, micro_batch_size: int, device: torch.device, in_memory: bool
    ) -> None:
        """Configure dataset."""
        self.micro_batch_size = micro_batch_size
        self.device = device

        dataloader = DataLoader(
            self.dataset,
            self.micro_batch_size,
            collate_fn=self.data_collator,
        )
        self.data_iter = iter(dataloader)

        def _inner_send_b2d(batch):
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)

        if not in_memory:
            self._send_batch_to_device = _inner_send_b2d
            return

        # do nothing in case of in-memory loading
        self._send_batch_to_device = lambda batch: None

        # load dataset into memory
        self.batches = []
        for batch in self.data_iter:
            _inner_send_b2d(batch)
            self.batches.append(batch)

        self.data_iter = iter(self.batches)

        logger.debug("loaded dataset into memory")

    def next_batch(self) -> Tensor | None:
        """Return next data tensor.

        Once all the data is consumed, it returns None.
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            return None

        # noop for in-memory case; otherwise, load batch to a correct device
        self._send_batch_to_device(batch)

        return batch

    def num_of_batches(self) -> int:
        """Return the number of batches from dataset.

        set_micro_batch_size() must be called before calling this function.
        """
        return math.ceil(len(self.dataset) / self.micro_batch_size)

    @staticmethod
    def create_image_dataset(
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str],
        split: Optional[str] = None,
    ) -> Tuple[Type[BaseImageProcessor], Dataset]:
        """Create image dataset."""
        # no need to load all datasets; since we need dataset for inference
        dataset = load_dataset(dataset_path, dataset_name, split=split)

        image_processor = AutoImageProcessor.from_pretrained(model_name)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )

        def transforms(example_batch):
            """Apply _transforms across a batch."""
            _normalize = Normalize(
                mean=image_processor.image_mean, std=image_processor.image_std
            )
            _transforms = Compose(
                [
                    Resize(size),
                    CenterCrop(size),
                    ToTensor(),
                    _normalize,
                ]
            )

            # TODO: image datasets have img or image as key;
            #       handling it in the following way can be error-prone
            #       need some automated way
            try:
                example_batch["pixel_values"] = [
                    _transforms(pil_img.convert("RGB"))
                    for pil_img in example_batch["img"]
                ]
            except KeyError:
                example_batch["pixel_values"] = [
                    _transforms(pil_img.convert("RGB"))
                    for pil_img in example_batch["image"]
                ]

            return example_batch

        dataset.set_transform(transforms)

        return image_processor, dataset

    @staticmethod
    def create_language_dataset(
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str],
        split: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ) -> Tuple[Type[PreTrainedTokenizerBase], Dataset]:
        """Create language dataset."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # no need to load all datasets; since we need a dataset for inference
        dataset = load_dataset(dataset_path, dataset_name, split=split)
        column_names = set(dataset.features)

        if "text" in column_names:
            text_column_name = "text"
        elif "prompt" in column_names:
            text_column_name = "prompt"
        else:
            text_column_name = column_names[0]

        if max_seq_length is None:
            max_seq_length = tokenizer.model_max_length

        def tokenize_function(examples):
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer(
                examples[text_column_name], padding=True, return_tensors="pt"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
        )

        return tokenizer, tokenized_dataset
