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

"""job.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from infscale import get_logger
from infscale.common.exceptions import InvalidConfig
from infscale.configs.controller import GenConfig


logger = None


@dataclass
class StageConfig:
    """Class for keeping stage information for worker."""

    start: int  # start layer number
    end: int  # end layer number
    id: str  # <stage number>-<replica number>


@dataclass
class StageData:
    """Class for keeping stage data for worker."""

    start: int  # start layer number
    end: int  # end layer number


@dataclass
class Dataset:
    """Specification about dataset.

    We only support hugggingface dataset currently.
    """

    path: str
    name: str
    split: str


@dataclass
class WorldInfo:
    """Specification about world info in the flow graph."""

    name: str
    peers: list[str]
    data_port: int = 30000
    ctrl_port: int = 30001
    addr: str = "127.0.0.1"
    backend: Optional[str] = ""


@dataclass
class WorkerData:
    """Specification about worker data."""

    id: str
    stage: StageData
    is_server: bool = False
    deploy: bool = False
    device: Optional[str] = ""


@dataclass
class ServeConfig:
    """Class for keeping config values of serve specification."""

    name: str

    model: str

    stage: StageConfig

    dataset: Dataset

    flow_graph: dict[str, list[WorldInfo]]

    workers_stage_info: dict[str, StageConfig]

    job_id: str

    reqgen_config: GenConfig

    device: str = "gpu"

    nfaults: int = 0  # no of faults to tolerate, default: 0 (no fault tolerance)

    micro_batch_size: int = 8

    fwd_policy: str = "random"

    # maximum number of requests in flight at any given point in time
    max_inflight: int = 1

    is_server: bool = False

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
                world_info = item if isinstance(item, WorldInfo) else WorldInfo(**item)
                self.flow_graph[k][i] = world_info

                if self.stage.id == k and world_info.backend == "nccl":
                    assert "cuda" in self.device, "nccl requires cuda device"

        self.reqgen_config = GenConfig(**self.reqgen_config)


@dataclass
class JobConfig:
    """Class for job config."""

    workers: list[WorkerData]
    name: str
    model: str
    flow_graph: dict[str, list[WorldInfo]]
    dataset: Dataset
    job_id: str
    nfaults: int = 0
    micro_batch_size: int = 8
    fwd_policy: str = "random"
    max_inflight: int = 1

    # this will be set by controller  based on its configuration
    reqgen_config: GenConfig | None = None

    def __post_init__(self) -> None:
        """Handle post init class variables."""
        for k in list(self.flow_graph.keys()):
            for i, item in enumerate(self.flow_graph[k]):
                world_info = item if isinstance(item, WorldInfo) else WorldInfo(**item)
                self.flow_graph[k][i] = world_info

        for j, w in enumerate(self.workers):
            worker = w if isinstance(w, WorkerData) else WorkerData(**w)
            self.workers[j] = worker

    def validate(self) -> None:
        """Validate job config."""
        worker_devices = {
            worker.id: worker.device.split(":")[0] for worker in self.workers
        }

        for wid, world_infos in self.flow_graph.items():
            for world_info in world_infos:
                self._validate_device_backend(wid, world_info, worker_devices)

    def _validate_device_backend(
        self, wid: str, world_info: WorldInfo, worker_devices: dict[str, str]
    ) -> None:
        """Validate device and backend values."""
        backend = world_info.backend

        if not backend:
            err_msg = "backend attribute is required when static policy is used"
            raise InvalidConfig(err_msg)

        device_type = worker_devices.get(wid, None)

        if not device_type:
            err_msg = "device attribute is required when static policy is used"
            raise InvalidConfig(err_msg)

        if backend not in {"gloo", "nccl"}:
            err_msg = f"unknown backend: {backend} for {wid}; Options: gloo, nccl"
            raise InvalidConfig(err_msg)

        if device_type == "cpu" and backend != "gloo":
            err_msg = f"invalid backend: {backend} for {wid}; choose gloo for cpu"
            raise InvalidConfig(err_msg)

    def get_serve_configs(self) -> list[ServeConfig]:
        """Convert job config into a list of serve config dict."""
        serve_configs = []
        global logger
        logger = get_logger()

        workers_stage_info = {}
        for w in self.workers:
            wid = w.id
            stage = w.stage
            workers_stage_info[wid] = {**stage, "id": wid}

        if self.max_inflight <= 0:
            logger.warning("max_inflight must be a positive number; using 1")
            self.max_inflight = 1

        for item in self.workers:
            if not item.deploy:
                continue

            config = {
                "name": self.name,
                "model": self.model,
                "flow_graph": self.flow_graph,
                "stage": {**item.stage, "id": item.id},
                "dataset": self.dataset,
                "nfaults": self.nfaults,
                "micro_batch_size": self.micro_batch_size,
                "fwd_policy": self.fwd_policy,
                "device": item.device,
                "workers_stage_info": workers_stage_info,
                "job_id": self.job_id,
                "max_inflight": self.max_inflight,
                "is_server": item.is_server,
                "reqgen_config": self.reqgen_config,
            }
            serve_configs.append(ServeConfig(**config))

        return serve_configs

    @staticmethod
    def is_identical(x: JobConfig, y: JobConfig) -> bool:
        """Determine if two job configs are structurally identical.

        This static method is used to determine whether a job config for update
        is the same as the currently deployed job config.
        Note that port numbers and other attributes (e.g., micro_batch_size)
        are currently ignored since reflecting such changes requires logic
        changes in the broad codebase (e.g., across controller, agent and worker)
        """
        if x is None or y is None:
            return False

        if len(x.workers) != len(y.workers):
            return False

        for xw, yw in zip(x.workers, y.workers):
            if xw != yw:
                return False

        if len(x.flow_graph) != len(y.flow_graph):
            return False

        for xitem, yitem in zip(x.flow_graph.items(), y.flow_graph.items()):
            (xkey, xworld_info_list) = xitem
            (ykey, yworld_info_list) = yitem
            if xkey != ykey:
                return False

            if len(xworld_info_list) != len(yworld_info_list):
                return False

            for xworld, yworld in zip(xworld_info_list, yworld_info_list):
                if (
                    xworld.addr != yworld.addr
                    or xworld.backend != yworld.backend
                    or xworld.name != yworld.name
                    or xworld.peers != yworld.peers
                ):
                    return False

        return True
