# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""plan.py."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Memory:
    """
    Represent memory usage for a stage.

    Attributes:
        static_memory_gb (float): The static memory used by the stage in gigabytes.
        dynamic_memory_gb (float): The dynamic memory used by the stage in gigabytes.
        total_memory_gb (float): The total memory used by the stage in gigabytes.
    """

    static_memory_gb: float
    dynamic_memory_gb: float
    total_memory_gb: float


@dataclass
class DataSizes:
    """
    Represent the data sizes for input and output for a stage.

    Attributes:
        input_size_mb (float): The input data size in megabytes.
        output_size_mb (float): The output data size in megabytes.
    """

    input_size_mb: float
    output_size_mb: float


@dataclass
class Stage:
    """
    Represent a single stage in the pipeline.

    Attributes:
        stage_id (int): The unique identifier for the stage.
        layer_range (list[int]): The range of layers processed in this stage.
        num_replicas (int): The number of replicas used in the stage.
        gpus (list[int]): The list of GPU IDs used by this stage.
        gpu_allocation (dict[int, int]): A dictionary mapping GPU IDs to their respective memory allocation.
        throughput (float): The throughput of this stage.
        forward_time_ms (float): The forward pass time for this stage in milliseconds.
        stage_time_ms (float): The total time the stage takes in milliseconds.
        memory (Memory): The memory usage details for the stage.
        data_sizes (DataSizes): The data size details for the stage.
    """

    stage_id: int
    layer_range: list[int]
    num_replicas: int
    gpus: list[int]
    gpu_allocation: dict[int, int]
    throughput: float
    forward_time_ms: float
    stage_time_ms: float
    memory: Memory
    data_sizes: DataSizes


@dataclass
class ExecPlan:
    """
    Represent the overall pipeline statistics, including details about batch processing, stages, and memory usage.

    Attributes:
        batch_size (int): The size of each batch in the pipeline.
        num_gpus_used (int): The number of GPUs used by the pipeline.
        total_latency (float): The total latency of the pipeline in milliseconds.
        throughput (float): The throughput of the pipeline.
        solving_time_for_batch (float): The solving time per batch in seconds.
        stages (list[Stage]): A list of stages in the pipeline.
        total_latency_ms (float): The total latency in milliseconds.
        pipeline_throughput (float): The throughput of the pipeline.
    """

    batch_size: int
    num_gpus_used: int
    total_latency: float
    throughput: float
    solving_time_for_batch: float
    stages: list[Stage]
    total_latency_ms: float
    pipeline_throughput: float

    @classmethod
    def from_json(cls, data: dict) -> ExecPlan:
        """
        Parse a JSON dictionary into an ExecPlan object.

        Args:
            data (dict): The JSON data as a dictionary to be parsed.

        Returns:
            ExecPlan: The parsed ExecPlan object.
        """
        stages = [
            Stage(
                stage_id=stage["stage_id"],
                layer_range=stage["layer_range"],
                num_replicas=stage["num_replicas"],
                gpus=stage["gpus"],
                gpu_allocation=stage["gpu_allocation"],
                throughput=stage["throughput"],
                forward_time_ms=stage["forward_time_ms"],
                stage_time_ms=stage["stage_time_ms"],
                memory=Memory(**stage["memory"]),
                data_sizes=DataSizes(**stage["data_sizes"]),
            )
            for stage in data["stages"]
        ]

        return cls(
            batch_size=data["batch_size"],
            num_gpus_used=data["num_gpus_used"],
            total_latency=data["total_latency"],
            throughput=data["throughput"],
            solving_time_for_batch=data["solving_time_for_batch"],
            stages=stages,
            total_latency_ms=data["total_latency_ms"],
            pipeline_throughput=data["pipeline_throughput"],
        )
