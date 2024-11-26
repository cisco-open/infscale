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
from infscale.controller.job_state import JobStateEnum

logger = get_logger()


@dataclass
class JobMetaData:
    """JobMetaData dataclass."""

    job_id: str
    config: JobConfig
    state: JobStateEnum
    wrkrs_to_start: set[str]  # workers to start
    wrkrs_to_update: set[str]  # workers to update
    wrkrs_to_stop: set[str]  # workers to stop


class JobManager:
    """JobManager class."""

    def __init__(self):
        """Initialize an instance."""
        self.jobs: dict[str, JobMetaData] = {}

    def process_config(self, config: JobConfig) -> None:
        """Process a config."""
        logger.debug(f"got new config: {config}")

        curr_config = None
        if config.job_id in self.jobs:
            curr_config = self.jobs[config.job_id].config

        results = self.compare_configs(curr_config, config)
        # updating config for exsiting workers will be handled by each worker
        wrkrs_to_start, wrkrs_to_update, wrkrs_to_stop = results

        if config.job_id in self.jobs:
            job_data = self.jobs[config.job_id]
            job_data.config = config
            job_data.state = JobStateEnum.UPDATING
            job_data.wrkrs_to_start = wrkrs_to_start
            job_data.wrkrs_to_update = wrkrs_to_update
            job_data.wrkrs_to_stop = wrkrs_to_stop
        else:
            job_data = JobMetaData(
                config.job_id,
                config,
                JobStateEnum.READY,
                wrkrs_to_start,
                wrkrs_to_update,
                wrkrs_to_stop,
            )
            self.jobs[config.job_id] = job_data

    def compare_configs(
        self, curr_config: JobConfig, new_config: JobConfig
    ) -> tuple[set[str], set[str], set[str]]:
        """Compare two flow_graph dictionaries, and return the diffs."""
        old_cfg = set(curr_config.flow_graph.keys()) if curr_config else set()
        new_cfg = set(new_config.flow_graph.keys())

        wrkrs_to_start = new_cfg - old_cfg
        wrkrs_to_stop = old_cfg - new_cfg

        wrkrs_to_update = set()
        for key in old_cfg & new_cfg:
            old_value = curr_config.flow_graph[key]
            new_value = new_config.flow_graph[key]

            if len(old_value) != len(new_value):
                wrkrs_to_update.add(key)
                continue

            for old_worker, new_worker in zip(old_value, new_value):
                old_peers = old_worker.peers

                # TODO: remove isinstance check when the config file is being
                #       sent through the api call
                if isinstance(new_worker, WorkerInfo):
                    new_peers = new_worker.peers
                else:
                    new_peers = new_worker["peers"]

                if old_peers != new_peers:
                    wrkrs_to_update.add(key)
                    break

        return wrkrs_to_start, wrkrs_to_update, wrkrs_to_stop

    def get_config(self, job_id: str) -> JobConfig | None:
        """Return a job config of given job name."""
        return self.jobs[job_id].config if job_id in self.jobs else None

    def get_workers(self, job_id: str, sort: str = "start") -> set[str]:
        """Return workers that match sort for a given job name.

        sort is one of the three values: start (default), update and stop.
        """
        if job_id not in self.jobs:
            return set()

        match sort:
            case "start":
                return self.jobs[job_id].wrkrs_to_start
            case "update":
                return self.jobs[job_id].wrkrs_to_update
            case "stop":
                return self.jobs[job_id].wrkrs_to_stop
            case _:
                raise ValueError(f"unknown sort: {sort}")
