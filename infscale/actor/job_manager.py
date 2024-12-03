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

from infscale import get_logger
from infscale.config import JobConfig, WorkerInfo
from infscale.controller.ctrl_dtype import JobAction
from infscale.controller.job_state import JobStateEnum

logger = None


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
        global logger
        logger = get_logger()

        self.jobs: dict[str, JobMetaData] = {}

    def process_config(self, config: JobConfig) -> None:
        """Process a config."""
        logger.debug(f"got new config: {config}")

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

    def compare_configs(
        self, curr_config: JobConfig, new_config: JobConfig
    ) -> tuple[set[str], set[str], set[str]]:
        """Compare two flow_graph dictionaries, and return the diffs."""
        old_cfg_wrkrs = set(curr_config.flow_graph.keys()) if curr_config else set()
        new_cfg_wrkrs = set(new_config.flow_graph.keys())

        start_wrkrs = new_cfg_wrkrs - old_cfg_wrkrs
        stop_wrkrs = old_cfg_wrkrs - new_cfg_wrkrs

        update_wrkrs = set()

        # select workers that will be affected by workers to be started
        for w, wrkr_info_list in new_config.flow_graph.items():
            for wrkr_info in wrkr_info_list:
                # TODO: remove isinstance check when the config file is being
                #       sent through the api call
                if isinstance(wrkr_info, WorkerInfo):
                    peers = wrkr_info.peers
                else:
                    peers = wrkr_info["peers"]

                self._pick_workers(update_wrkrs, start_wrkrs, w, peers)

        if curr_config is None:
            return start_wrkrs, update_wrkrs, stop_wrkrs

        # select workers that will be affected by workers to be stopped
        for w, wrkr_info_list in curr_config.flow_graph.items():
            for wrkr_info in wrkr_info_list:
                # TODO: remove isinstance check when the config file is being
                #       sent through the api call
                if isinstance(wrkr_info, WorkerInfo):
                    peers = wrkr_info.peers
                else:
                    peers = wrkr_info["peers"]

                self._pick_workers(update_wrkrs, stop_wrkrs, w, peers)

        return start_wrkrs, update_wrkrs, stop_wrkrs

    def _pick_workers(
        self,
        res_set: set[str],
        needles: set[str],
        name: str,
        peers: list[str],
    ) -> None:
        """Pick workers to update given needles and haystack.

        The needles are workers to start or stop and the haystack is
        name and peers.
        """
        if name in needles:  # in case name is in the needles
            for peer in peers:
                if peer in needles:
                    # if peer is also in the needles,
                    # the peer is not the subject of update
                    # because it is a worker that we start or stop
                    continue
                res_set.add(peer)

        else:  # in case name is not in the needles
            for peer in peers:
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

    def get_config(self, job_id: str) -> JobConfig | None:
        """Return a job config of given job name."""
        return self.jobs[job_id].config if job_id in self.jobs else None

    def get_workers(self, job_id: str, sort: JobAction = JobAction.START) -> set[str]:
        """Return workers that match sort for a given job name."""
        if job_id not in self.jobs:
            return set()

        # TODO: in order to avoid creation of similar enum class,
        #       we repurpose JobAction as argument to decide how to filter
        #       workers. This is not ideal because the purpose of JobAction
        #       is different from the usage in this method.
        #       we eed to revisit this later.
        match sort:
            case JobAction.START:
                return self.jobs[job_id].start_wrkrs
            case JobAction.UPDATE:
                return self.jobs[job_id].update_wrkrs
            case JobAction.STOP:
                return self.jobs[job_id].stop_wrkrs
            case _:
                raise ValueError(f"unknown sort: {sort}")
