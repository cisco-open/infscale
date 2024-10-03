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
# https://github.com/SymbioticLab/Oobleck/blob/3b7a0c2f19bff0991e623ffbeb8a5b365853bf3a/oobleck/module/model.py
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Union

import torch
from infscale import get_logger
from infscale.module.sharder import Sharder

if TYPE_CHECKING:
    from infscale.module.model_metadata import BaseModelMetaData

RANDOM_SEED = 42

logger = get_logger()


class ModelIR:
    """
    An intermediate representation class for HuggingFace model.

    HuggingFace models are downloaded from Hugging Face Hub
    (https://huggingface.co/models).

    It runs huggingface.utils.fx.symbolic_trace to get GraphModule
    and shard it to multiple GraphModules for pipeline execution.

    Model initialization must be done before distributed initialization.
    """

    def __init__(self, mmd: BaseModelMetaData):
        """Initialize the class."""
        # Initialize CPU seed
        random.seed(RANDOM_SEED)
        torch.default_generator.manual_seed(RANDOM_SEED)

        self.mmd = mmd

        self.layers = Sharder.shard(mmd)
        self.model_name = mmd.name

        self.total_num_params = sum(
            sum(p.numel() for p in layer.parameters()) for layer in self.layers
        )

        self.model_args = mmd.config
        self.output_parser: Union[Callable, None] = mmd.get_output_parser()
        self.predict_fn: Union[Callable, None] = mmd.get_predict_fn()

        logger.debug(f"# of layers = {len(self.layers)}")
