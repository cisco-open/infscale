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

from infscale import get_logger

logger = get_logger(__name__)


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
