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

from infscale.configs.job import JobConfig
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

        results = JobConfig.categorize_workers(curr_config, config)
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
