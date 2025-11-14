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

import copy
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
    recover: bool = False
    conflict_count: int = 0


@dataclass
class WorkerData:
    """Specification about worker data."""

    id: str
    stage: StageData
    is_server: bool = False
    deploy: bool = False
    device: Optional[str] = ""
    recover: bool = False


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

    force_terminate: bool = False

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

    def kv_cache_needed(self) -> bool:
        """Return if kv cache is necessary for serving."""
        return "llama" in self.model.lower()

    @staticmethod
    def get_worlds_to_configure(
        curr_spec: ServeConfig, new_spec: ServeConfig
    ) -> set[str]:
        """Compare two specs and return new and updated worlds."""
        helper = ServeConfigHelper()

        new_worlds = helper._get_worlds(new_spec)
        new_world_names = set(new_worlds.keys())

        # if current spec is not available,
        # return worlds from the new spec.
        if curr_spec is None:
            return new_world_names

        curr_worlds = helper._get_worlds(curr_spec)
        curr_world_names = set(curr_worlds.keys())

        deploy_worlds = new_world_names - curr_world_names

        common_keys = curr_world_names & new_world_names

        updated_worlds = {
            k
            for k in common_keys
            if (
                curr_worlds[k].addr != new_worlds[k].addr
                or curr_worlds[k].data_port != new_worlds[k].data_port
                or curr_worlds[k].ctrl_port != new_worlds[k].ctrl_port
            )
        }

        return deploy_worlds | updated_worlds


class ServeConfigHelper:
    """Class for defining helper methods for serve config."""

    def _get_worlds(self, spec: ServeConfig) -> dict[str, WorldInfo]:
        """Return world names that relates to worker id."""
        id = spec.stage.id

        return {
            world_info.name: world_info
            for wrk_id, worlds in spec.flow_graph.items()
            for world_info in worlds
            if id == wrk_id or id in world_info.peers
        }


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
    recover: bool = True
    force_terminate: bool = False

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

    def reset_recover_flags(self) -> None:
        """Reset recover flags on world and worker."""
        for worker in self.workers:
            worker.recover = False

        for world_list in self.flow_graph.values():
            for world in world_list:
                world.recover = False

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
                "force_terminate": self.force_terminate,
            }
            serve_configs.append(ServeConfig(**config))

        return serve_configs

    def max_stage_id(self) -> int:
        """Return the maximum stage id from config."""
        max_id = 0

        for worker in self.workers:
            if worker.id.startswith("s"):
                continue

            stage_id = int(worker.id.split("-")[0])
            max_id = max(max_id, stage_id)

        return max_id

    def max_world_id(self) -> int:
        """Return the maximum world id from config."""
        max_id = 0

        for world_info_list in self.flow_graph.values():
            for world_info in world_info_list:
                # we assume that the prefix is a single character (e.g., w)
                world_id = int(world_info.name[1:])
                max_id = max(max_id, world_id)

        return max_id

    def server_id(self) -> str | None:
        """Return server id."""
        for worker in self.workers:
            if worker.is_server:
                return worker.id

        return None

    def server_ip(self) -> str:
        """Return IP address of server."""
        server_id = self.server_id()
        if server_id is None:
            return ""

        return self.flow_graph[server_id][0].addr

    def is_auto_regressive(self) -> bool:
        """Return if model has auto regressive nature for serving."""
        return "llama" in self.model.lower()

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

    @staticmethod
    def world_name(world_id: int) -> str:
        """Return world name given a world id."""
        return f"w{world_id}"

    @staticmethod
    def get_pipeline_identifiers(new_cfg: JobConfig) -> set[str]:
        """Get pipeline identifiers based on server id."""
        server_id = new_cfg.server_id()

        wrk_ids = set()

        for wid, worlds_list in new_cfg.flow_graph.items():
            for world_info in worlds_list:
                if server_id in world_info.peers:
                    wrk_ids.add(wid)

        return wrk_ids

    @staticmethod
    def merge(base: JobConfig, extra: JobConfig) -> JobConfig:
        """Merge two job configs and create a merged job config."""
        if base is None:
            return extra

        cfg = copy.deepcopy(base)
        for worker_id, world_info_list in extra.flow_graph.items():
            if worker_id not in cfg.flow_graph:
                cfg.flow_graph[worker_id] = world_info_list
            else:
                tmp = cfg.flow_graph[worker_id]
                cfg.flow_graph[worker_id] = tmp + world_info_list

        cfg.workers = cfg.workers + extra.workers

        return cfg

    @staticmethod
    def categorize_workers(
        curr_config: JobConfig, new_config: JobConfig
    ) -> tuple[set[str], set[str], set[str]]:
        """Compare two flow_graph dictionaries, and return the diffs."""
        helper = JobConfigHelper()
        old_cfg_wrkrs = set(curr_config.flow_graph.keys()) if curr_config else set()
        new_cfg_wrkrs = set(new_config.flow_graph.keys())
        recover_wrkrs = helper.get_recover_worker_ids(new_config)

        start_wrkrs = recover_wrkrs | (new_cfg_wrkrs - old_cfg_wrkrs)
        stop_wrkrs = old_cfg_wrkrs - new_cfg_wrkrs

        update_wrkrs = set()

        # select workers that will be affected by workers to be started
        for w, world_info_list in new_config.flow_graph.items():
            for new_world_info in world_info_list:
                curr_world_info = helper.find_matching_world_info(
                    curr_config, w, new_world_info
                )
                helper.pick_workers(
                    update_wrkrs, start_wrkrs, w, new_world_info, curr_world_info
                )

        if curr_config is None:
            return start_wrkrs, update_wrkrs, stop_wrkrs

        # select workers that will be affected by workers to be stopped
        for w, world_info_list in curr_config.flow_graph.items():
            for new_world_info in world_info_list:
                curr_world_info = helper.find_matching_world_info(
                    curr_config, w, new_world_info
                )
                helper.pick_workers(
                    update_wrkrs, stop_wrkrs, w, new_world_info, curr_world_info
                )

        # due to pervious state, recover workers are included in update workers
        # therefore, recover workers need to be removed from the updated ones.
        update_wrkrs -= recover_wrkrs

        return start_wrkrs, update_wrkrs, stop_wrkrs

    @staticmethod
    def get_workers_diff(a: JobConfig, b: JobConfig) -> set[str]:
        """Return a set of worker ids diffs based on old and new cfg."""
        old_workers = {worker.id for worker in a.workers}
        new_workers = {worker.id for worker in b.workers}

        return old_workers - new_workers

    @staticmethod
    def remove_pipeline(config: JobConfig, workers_to_remove: set[str]) -> JobConfig:
        """Remove pipelines from config based on workers to remove."""
        cfg = copy.deepcopy(config)
        helper = JobConfigHelper()

        # 1: find server id
        server_id = helper.get_server_id(config)

        # 2: find all workers in the pipeline starting from workers_to_remove
        to_remove = helper.find_pipeline_nodes(
            cfg.flow_graph, workers_to_remove, server_id
        )

        # 3: remove workers from flow_graph
        for wid in to_remove:
            cfg.flow_graph.pop(wid, None)

        # 4: remove WorldInfo entries in ALL remaining workers
        #    if their peers reference any removed worker
        for wid, worlds in list(cfg.flow_graph.items()):
            cfg.flow_graph[wid] = [
                w for w in worlds if all(peer not in to_remove for peer in w.peers)
            ]

        # 5: remove loop workers from workers list
        cfg.workers = [w for w in cfg.workers if w.id not in to_remove]

        return cfg


class JobConfigHelper:
    """Class for defining helper methods for job config."""

    def get_server_id(self, config: JobConfig) -> str:
        return next((w.id for w in config.workers if w.is_server), "")

    def find_pipeline_nodes(
        self,
        flow_graph: dict[str, list[WorldInfo]],
        workers_to_remove: set[str],
        server_id: str,
    ) -> set[str]:
        """
        Remove workers that are either:
        - failed themselves, or
        - no longer part of a live pipeline (cannot reach server anymore).

        Uses DFS in both forward and reverse directions.
        """

        # build forward and reverse graphs
        fwd = {wid: [] for wid in flow_graph}
        rev = {wid: [] for wid in flow_graph}

        for wid, worlds in flow_graph.items():
            for w in worlds:
                for peer in w.peers:
                    fwd.setdefault(peer, []).append(wid)
                    rev.setdefault(wid, []).append(peer)

        # remove failed nodes from graph
        for failed in workers_to_remove:
            fwd.pop(failed, None)
            rev.pop(failed, None)

        # DFS1: reachable from server
        reachable_from_server = set()
        stack = [server_id]

        while stack:
            node = stack.pop()
            if node in reachable_from_server:
                continue
            if node in workers_to_remove:
                continue
            reachable_from_server.add(node)
            for nxt in fwd.get(node, []):
                if nxt not in workers_to_remove:
                    stack.append(nxt)

        # DFS2: can reach to server
        can_reach_server = set()
        stack = [server_id]

        while stack:
            node = stack.pop()
            if node in can_reach_server:
                continue
            if node in workers_to_remove:
                continue
            can_reach_server.add(node)
            for prev in rev.get(node, []):
                if prev not in workers_to_remove:
                    stack.append(prev)

        # surviving nodes: intersection
        survivors = reachable_from_server & can_reach_server

        # everything else (except server) is removed
        to_remove = {
            wid for wid in flow_graph if wid != server_id and wid not in survivors
        }

        return to_remove

    def get_recover_worker_ids(self, config: JobConfig) -> set[str]:
        """Return a set of worker IDs that need to be recovered."""
        return {worker.id for worker in config.workers if worker.recover}

    def pick_workers(
        self,
        res_set: set[str],
        needles: set[str],
        name: str,
        new_world_info: WorldInfo,
        curr_world_info: WorldInfo | None,
    ) -> None:
        """Pick workers to update given needles and haystack.

        The needles are workers to start or stop and the haystack is
        name and peers.

        Also includes peers of `name` if its connection details
        (`addr`, `ctrl_port`, `data_port`) differ from the previous config.
        """
        if curr_world_info and self.has_connection_changed(
            curr_world_info, new_world_info
        ):
            for peer in new_world_info.peers:
                res_set.add(peer)

        if name in needles:  # in case name is in the needles
            for peer in new_world_info.peers:
                if peer in needles:
                    # if peer is also in the needles,
                    # the peer is not the subject of update
                    # because it is a worker that we start or stop
                    continue

                res_set.add(peer)

        else:  # in case name is not in the needles
            for peer in new_world_info.peers:
                if peer not in needles:
                    continue

                # if peer is in the needles,
                # the peer is a worker that we start or stop
                # so, name is a subect of update
                # because name is affected by the peer

                res_set.add(name)

                # we don't need to check other peers
                # because name is already affected by one peer
                # so we come out of the for-loop
                break

    def has_connection_changed(self, old: WorldInfo, new: WorldInfo) -> bool:
        """Check if worker connection details are changed."""
        return (
            old.addr != new.addr
            or old.ctrl_port != new.ctrl_port
            or old.data_port != new.data_port
        )

    def find_matching_world_info(
        self, curr_config: JobConfig | None, w: str, new_world_info: WorldInfo
    ) -> WorldInfo | None:
        """Return current world info or None if there is no current config."""
        if not curr_config:
            return None

        for curr_info in curr_config.flow_graph.get(w, []):
            if curr_info.name == new_world_info.name:
                return curr_info

        return None
