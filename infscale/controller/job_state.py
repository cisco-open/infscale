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

"""AgentContext class."""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional, Tuple

from infscale.controller.apiserver import JobAction


class JobStateEnum(Enum):
    """JobState enum."""

    STARTED = "started"
    STARTING = "starting"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UPDATING = "updating"


JOB_ALLOWED_ACTIONS = MappingProxyType(
    {
        JobStateEnum.STARTED: (JobAction.STOP, JobAction.UPDATE),
        JobStateEnum.STARTING: (JobAction.STOP),
        JobStateEnum.STOPPED: (JobAction.START),
        JobStateEnum.STOPPING: (),
        JobStateEnum.UPDATING: (JobAction.STOP),
    }
)


@dataclass
class JobStateData:
    """WorkerMetaData dataclass."""

    state: JobStateEnum
    possible_actions: Optional[Tuple[JobAction, ...]]


class JobState:
    """JobState class."""

    def __init__(self):
        """Initialize instance."""
        self.job_status: dict[str, dict[str, JobStateData]] = dict()

    def remove_job(self, job_id: str) -> None:
        agent_ids = self._get_job_agent_ids(job_id)

        for agent_id in agent_ids:
            # remove job from agent dict for now
            del self.job_status[agent_id][job_id]

    def set_agent(self, agent_id: str) -> None:
        """Sets agent ID in job status dict"""

        self.job_status[agent_id] = dict()

    def _is_new_job(self, job_id: str, job_action: JobAction) -> bool:
        return self._get_job_agent_ids(job_id) is None and job_action == JobAction.START

    def set_job_state(
        self, job_id: str, job_action: JobAction, job_state: JobStateEnum = None
    ) -> None:
        """Updates job state"""

        new_job = self._is_new_job(job_id, job_action)

        if new_job:
            agent_id = self._get_available_agent_id()
            
            self.job_status[agent_id][job_id] = JobStateData(
                JobStateEnum.STARTING, JOB_ALLOWED_ACTIONS.get(JobStateEnum.STARTING)
            )

            return
        
        agent_ids = self._get_job_agent_ids(job_id)

        for agent_id in agent_ids:
            self.job_status[agent_id][job_id] = JobStateData(
                job_state, JOB_ALLOWED_ACTIONS.get(job_state)
            )

    def _get_available_agent_id(self) -> str:
        """Finds agent with less workload"""

        return min(self.job_status, key=lambda agent_id: len(self.job_status[agent_id]))

    def can_update_job_state(self, job_id: str, job_action: JobAction) -> bool:
        """Checks if an update is possible based on job state"""
        if self._is_new_job(job_id, job_action):
            return True

        agent_ids = self._get_job_agent_ids(job_id)

        can_update = False

        for agent_id in agent_ids:
            if job_action in self.job_status[agent_id][job_id].possible_actions:
                can_update = True
    
        return can_update

    def _get_job_agent_ids(self, job_id) -> str | None:
        """Returns agent_ids for job_id or None"""
        agent_ids = []
        for agent_id, jobs in self.job_status.items():
            if job_id in jobs:
                agent_ids.append(agent_id)  # Return agent_id if the job is already present

        if len(agent_ids):
            return agent_ids

        return None  # Return None if the job_id was not found
