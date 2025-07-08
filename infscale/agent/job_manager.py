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

"""job_manager.py."""

from dataclasses import dataclass

from infscale.configs.job import JobConfig, WorldInfo
from infscale.controller.ctrl_dtype import CommandAction
from infscale.controller.job_context import JobStateEnum


@dataclass
class JobMetaData:
    """JobMetaData dataclass."""

    job_id: str
    config: JobConfig
    state: JobStateEnum
    start_wrkrs: set[str]  # workers to start
    update_wrkrs: set[str]  # workers to update
    stop_wrkrs: set[str]  # workers to stop


class JobManager:
    """JobManager class."""

    def __init__(self):
        """Initialize an instance."""
        self.jobs: dict[str, JobMetaData] = {}

    def cleanup(self, job_id: str) -> None:
        """Remove job related data."""
        del self.jobs[job_id]

    def get_job_data(self, job_id) -> JobMetaData:
        """Get JobMetaData of a given job id."""
        return self.jobs[job_id]

    def process_config(self, config: JobConfig) -> None:
        """Process a config."""

        curr_config = None
        if config.job_id in self.jobs:
            curr_config = self.jobs[config.job_id].config

        results = self.compare_configs(curr_config, config)
        # updating config for exsiting workers will be handled by each worker
        start_wrkrs, update_wrkrs, stop_wrkrs = results

        if config.job_id in self.jobs:
            job_data = self.jobs[config.job_id]
            job_data.config = config
            job_data.state = JobStateEnum.UPDATING
            job_data.start_wrkrs = start_wrkrs
            job_data.update_wrkrs = update_wrkrs
            job_data.stop_wrkrs = stop_wrkrs
        else:
            job_data = JobMetaData(
                config.job_id,
                config,
                JobStateEnum.READY,
                start_wrkrs,
                update_wrkrs,
                stop_wrkrs,
            )
            self.jobs[config.job_id] = job_data
            
    def _get_recover_worker_ids(self, config: JobConfig) -> set[str]:
        """Return a set of worker IDs that need to be recovered."""
        wrk_ids = {worker.id for worker in config.workers if worker.recover}
        
        return wrk_ids

    def compare_configs(
        self, curr_config: JobConfig, new_config: JobConfig
    ) -> tuple[set[str], set[str], set[str]]:
        """Compare two flow_graph dictionaries, and return the diffs."""
        old_cfg_wrkrs = set(curr_config.flow_graph.keys()) if curr_config else set()
        new_cfg_wrkrs = set(new_config.flow_graph.keys())

        recover_wrkrs = self._get_recover_worker_ids(new_config)

        start_wrkrs = recover_wrkrs | (new_cfg_wrkrs - old_cfg_wrkrs)
        stop_wrkrs = old_cfg_wrkrs - new_cfg_wrkrs

        update_wrkrs = set()

        # select workers that will be affected by workers to be started
        for w, world_info_list in new_config.flow_graph.items():
            for new_world_info in world_info_list:
                curr_world_info = self._find_matching_world_info(curr_config, w, new_world_info)
                self._pick_workers(update_wrkrs, start_wrkrs, w, new_world_info, curr_world_info)

        if curr_config is None:
            return start_wrkrs, update_wrkrs, stop_wrkrs

        # select workers that will be affected by workers to be stopped
        for w, world_info_list in curr_config.flow_graph.items():
            for new_world_info in world_info_list:
                curr_world_info = self._find_matching_world_info(curr_config, w, new_world_info)
                self._pick_workers(update_wrkrs, stop_wrkrs, w, new_world_info, curr_world_info)

        return start_wrkrs, update_wrkrs, stop_wrkrs

    def _pick_workers(
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
        if curr_world_info and self._has_connection_changed(curr_world_info, new_world_info):
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
            
    def _has_connection_changed(
        self, old: WorldInfo, new: WorldInfo
    ) -> bool:
        """Check if worker connection details are changed."""
        return (
            old.addr != new.addr or
            old.ctrl_port != new.ctrl_port or
            old.data_port != new.data_port
        )
            
    def _find_matching_world_info(
        self, curr_config: JobConfig | None, w: str, new_world_info: WorldInfo
    ) -> WorldInfo | None:
        """Return current world info or None if there is no current config."""
        if not curr_config:
            return None

        for curr_info in curr_config.flow_graph.get(w, []):
            if curr_info.name == new_world_info.name:
                return curr_info

        return None

    def get_config(self, job_id: str) -> JobConfig | None:
        """Return a job config of given job name."""
        return self.jobs[job_id].config if job_id in self.jobs else None

    def get_workers(
        self, job_id: str, sort: CommandAction = CommandAction.START
    ) -> set[str]:
        """Return workers that match sort for a given job name."""
        if job_id not in self.jobs:
            return set()

        # TODO: in order to avoid creation of similar enum class,
        #       we repurpose CommandAction as argument to decide how to filter
        #       workers. This is not ideal because the purpose of CommandAction
        #       is different from the usage in this method.
        #       we eed to revisit this later.
        match sort:
            case CommandAction.START:
                return self.jobs[job_id].start_wrkrs
            case CommandAction.UPDATE:
                return self.jobs[job_id].update_wrkrs
            case CommandAction.STOP:
                return self.jobs[job_id].stop_wrkrs
            case _:
                raise ValueError(f"unknown sort: {sort}")
