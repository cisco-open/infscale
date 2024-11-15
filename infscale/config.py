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

"""Config parser."""

from dataclasses import dataclass
from typing import Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr

from infscale import get_logger

logger = get_logger(__name__)

RAW_KEY_PARTITIONS = "partitions"
RAW_KEY_MINI_BATCH_SIZE = "mini_batch_size"
RAW_KEY_PRE_TRAINED = "pre_trained"
RAW_KEY_DEVICES = "devices"
RAW_KEY_REPETITION = "repetition"

RAW_KEY_INDEX = "index"
RAW_KEY_NUM_SHARDS = "num_shards"

DEFAULT_MINI_BATCH_SIZE = 8


class Partitions(BaseModel):
    """Partitions class."""

    index_shards_map: dict

    _indexes: list = PrivateAttr([])
    _num_shards: int = PrivateAttr(-1)
    _name: str = PrivateAttr(None)

    def get_all(self):
        """Return all pairs of parition index and its shards."""
        index_shards_pairs = []
        for index in sorted(self.index_shards_map.keys()):
            num_shards = self.index_shards_map[index]
            index_shards_pairs.append((index, num_shards))

        return index_shards_pairs

    def get_partition_indexes(self) -> list[int]:
        """Return indexes of partitions."""
        if len(self._indexes) > 0:
            return self._indexes

        self._indexes = list(sorted(self.index_shards_map.keys()))

        return self._indexes

    def get_num_shards(self) -> int:
        """Return the total number of shards in partitions."""
        if self._num_shards > 0:
            return self._num_shards

        self._num_shards = 0
        for _, v in self.index_shards_map.items():
            self._num_shards += v

        return self._num_shards

    def get_name(self) -> str:
        """Return name of the partitions.

        The name is composed with index and number of shards.
        """
        if self._name:
            return self._name

        subnames = []
        for index in sorted(self.index_shards_map.keys()):
            num_shards = self.index_shards_map[index]
            subnames.append(f"{index}:{num_shards}")

        self._name = "-".join(subnames)

        return self._name


class Config(BaseModel):
    """Config class."""

    def __init__(self, config_path: str):
        """Initialize class instance."""
        raw_config = read_config(config_path)
        transformed_config = transform_config(raw_config)

        super().__init__(**transformed_config)

    partitions: Partitions
    mini_batch_size: Optional[int] = Field(default=DEFAULT_MINI_BATCH_SIZE)
    pre_trained: Optional[bool] = Field(defaut=False)
    devices: Optional[list[str]] = Field(default=[])
    repetition: Optional[int] = Field(default=1)


def read_config(filename: str) -> dict:
    """Read YAML format config."""
    with open(filename) as f:
        return yaml.safe_load(f)


def transform_config(raw_config: dict) -> dict:
    """Transform config."""
    mini_batch_size = raw_config.get(RAW_KEY_MINI_BATCH_SIZE, DEFAULT_MINI_BATCH_SIZE)
    pre_trained = raw_config.get(RAW_KEY_PRE_TRAINED, False)

    devices = raw_config.get(RAW_KEY_DEVICES, [])
    repetition = raw_config.get(RAW_KEY_REPETITION, 1)

    index_shards_map = transform_partitions(raw_config[RAW_KEY_PARTITIONS])

    config_data = {
        RAW_KEY_MINI_BATCH_SIZE: mini_batch_size,
        RAW_KEY_PRE_TRAINED: pre_trained,
        RAW_KEY_DEVICES: devices,
        RAW_KEY_REPETITION: repetition,
        RAW_KEY_PARTITIONS: index_shards_map,
    }

    return config_data


def transform_partitions(raw_partitions_config: dict):
    """Transform partitions into kv pairs."""
    index_zero_found = False

    index_shards_map = {}
    for raw_index_shards in raw_partitions_config:
        index = raw_index_shards[RAW_KEY_INDEX]
        shards = raw_index_shards[RAW_KEY_NUM_SHARDS]

        if index == 0:
            index_zero_found = True

        if index in index_shards_map:
            raise ValueError(f"Duplicate index {index} specified")

        # TODO: 0 can be used to indicatre dynamic scaling
        if shards <= 0:
            print("WARNING: The number of shards can't be less than 0")
            print("         The value is set to 1")
            shards = 1
        index_shards_map[index] = shards

    if not index_zero_found:
        raise ValueError("config the first partition (index 0) not found")

    return Partitions(index_shards_map=index_shards_map)


"""
TODO: The following is temporary serving config dataclasses.
      They should be revised over time.
"""


@dataclass
class StageConfig:
    """Class for keeping stage information for worker."""

    start: int  # start layer number
    end: int  # end layer number
    id: str  # <stage number>-<replica number>
    is_server: bool = False


@dataclass
class Dataset:
    """Specification about dataset.

    We only support hugggingface dataset currently.
    """

    path: str
    name: str
    split: str


@dataclass
class WorkerInfo:
    """Specification about worker info in the flow graph."""

    name: str
    addr: str
    port: int
    peers: list[str]
    backend: str = "gloo"


@dataclass
class ServeConfig:
    """Class for keeping config values of serve specification."""

    name: str

    model: str

    stage: StageConfig

    dataset: Dataset

    flow_graph: dict[str, list[WorkerInfo]]

    rank_map: dict[str, int]

    workers_stage_info: dict[str, StageConfig]

    job_id: str

    device: str = "cpu"

    nfaults: int = 0  # no of faults to tolerate, default: 0 (no fault tolerance)

    micro_batch_size: int = 8

    fwd_policy: str = "random"

    # maximum number of requests in flight at any given point in time
    max_inflight: int = 1

    def __post_init__(self):
        """Convert stage dict into stage object."""
        # TODO - remove isinstance check when the config file is being sent through the api call
        self.dataset = (
            self.dataset
            if isinstance(self.dataset, Dataset)
            else Dataset(**self.dataset)
        )
        self.stage = StageConfig(**self.stage)

        for k in list(self.workers_stage_info.keys()):
            stage = self.workers_stage_info[k]
            self.workers_stage_info[k] = (
                stage if isinstance(stage, StageConfig) else StageConfig(**stage)
            )

        for k in list(self.flow_graph.keys()):
            for i, item in enumerate(self.flow_graph[k]):
                worker_info = (
                    item if isinstance(item, WorkerInfo) else WorkerInfo(**item)
                )
                self.flow_graph[k][i] = worker_info

                if worker_info.backend == "nccl":
                    assert "cuda" in self.device, "nccl requires cuda device"


@dataclass
class JobConfig:
    """Class for job config."""

    workers: list
    name: str
    model: str
    flow_graph: dict[str, list[WorkerInfo]]
    rank_map: dict[str, int]
    dataset: Dataset
    job_id: str
    nfaults: int = 0
    micro_batch_size: int = 8
    fwd_policy: str = "random"
    max_inflight: int = 1

    def get_serve_configs(self) -> list[ServeConfig]:
        """Convert job config into a list of serve config dict."""
        serve_configs = []

        workers_stage_info = {}
        for worker in self.workers:
            wid = worker["id"]
            stage = worker["stage"]
            workers_stage_info[wid] = {**stage, "id": wid}

        if self.max_inflight <= 0:
            logger.warn("max_inflight must be a positive number; using 1")
            self.max_inflight = 1

        for item in self.workers:
            config = {
                "name": self.name,
                "model": self.model,
                "flow_graph": self.flow_graph,
                "stage": {**item["stage"], "id": item["id"]},
                "rank_map": self.rank_map,
                "dataset": self.dataset,
                "nfaults": self.nfaults,
                "micro_batch_size": self.micro_batch_size,
                "fwd_policy": self.fwd_policy,
                "device": item["device"],
                "workers_stage_info": workers_stage_info,
                "job_id": self.job_id,
                "max_inflight": self.max_inflight,
            }
            serve_configs.append(ServeConfig(**config))

        return serve_configs
