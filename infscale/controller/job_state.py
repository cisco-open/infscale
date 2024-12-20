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

"""JobState class."""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Optional, Tuple

from infscale.config import JobConfig
from infscale.controller.ctrl_dtype import JobAction


class JobStateEnum(Enum):
    """JobState enum."""

    READY = "ready"
    STARTED = "started"
    STARTING = "starting"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UPDATING = "updating"


JOB_ALLOWED_ACTIONS = MappingProxyType(
    {
        JobStateEnum.READY: (JobAction.START),
        JobStateEnum.STARTED: (JobAction.STOP, JobAction.UPDATE),
        # TODO: we add JobAction.UPDATE for STARTING; this is a hack
        #       to allow job update because the state doesn't transition
        #       into STARTED; overall the state management code needs
        #       to be refactored so that the state mamangement becomes
        #       more intuitive.
        JobStateEnum.STARTING: (JobAction.STOP, JobAction.UPDATE),
        JobStateEnum.STOPPED: (JobAction.START),
        JobStateEnum.STOPPING: (),
        JobStateEnum.UPDATING: (JobAction.STOP),
    }
)


@dataclass
class JobStateData:
    """JobStateData dataclass."""

    state: JobStateEnum
    config: JobConfig = None
    num_new_workers: int = 0
    new_config: JobConfig = None
    ports: list[int] = None


class JobState:
    """JobState class."""

    def __init__(self):
        """Initialize instance."""
        self.job_status: dict[str, dict[str, JobStateData]] = dict()

    def remove_job(self, job_id: str) -> None:
        """Remove job ID."""
        agent_ids = self._get_job_agent_ids(job_id)

        for agent_id in agent_ids:
            # remove job from agent dict for now
            del self.job_status[agent_id][job_id]

    def set_agent(self, agent_id: str) -> None:
        """Set agent ID in job status dict."""
        self.job_status[agent_id] = dict()

    def job_exists(self, job_id: str) -> bool:
        """Check whether job exists or not."""
        for jobs in self.job_status.values():
            if job_id in jobs:
                return True

        return False

    def get_config(self, agent_id: str, job_id: str) -> JobConfig:
        """Get job config or raise ValueError."""
        if not self._has_config(agent_id, job_id):
            raise ValueError(f"no config available for job: {job_id}")

        return self.job_status[agent_id][job_id].config

    def process_cfg(self, agent_id: str, job_id: str, new_cfg: JobConfig) -> None:
        """Process received config from controller."""
        job_state = self.get_job_state(agent_id, job_id)

        job_state.new_config = new_cfg
        job_state.num_new_workers = self._count_new_workers(job_state)

    def _count_new_workers(self, job_state: JobStateData) -> int:
        """Return the number of new workers in new config."""
        old_cfg, new_cfg = job_state.config, job_state.new_config

        curr_workers = []
        if old_cfg is not None:
            graph_values = old_cfg.flow_graph.values()
            curr_workers = [w.name for w_list in graph_values for w in w_list]

        graph_values = new_cfg.flow_graph.values()
        new_workers = [w.name for w_list in graph_values for w in w_list]

        return len(set(new_workers) - set(curr_workers))

    def get_job_state(self, agent_id: str, job_id: str) -> JobStateData:
        """Return the state of a job."""
        return self.job_status[agent_id][job_id]

    def set_ports(self, agent_id: str, job_id: str, ports: list[int]) -> None:
        """Set ports for a job."""
        state = self.get_job_state(agent_id, job_id)

        state.ports = ports

    def _has_config(self, agent_id: str, job_id: str) -> bool:
        if agent_id not in self.job_status:
            return False

        if job_id not in self.job_status[agent_id]:
            return False

        state = self.get_job_state(agent_id, job_id)

        return state.config is not None

    def set_job_state(
        self,
        agent_id: str,
        job_id: str,
        job_state: JobStateEnum = JobStateEnum.READY,
    ) -> None:
        """Update job state."""
        agent_ids = self._get_job_agent_ids(job_id)

        # new job
        if agent_ids is None:
            self.job_status[agent_id][job_id] = JobStateData(
                job_state,
            )

            return

        for agent_id in agent_ids:
            state = self.get_job_state(agent_id, job_id)
            state.state = job_state

    def _get_available_agent_id(self) -> str | None:
        """Find agent with less workload."""
        if len(self.job_status) == 0:
            return None

        return min(self.job_status, key=lambda agent_id: len(self.job_status[agent_id]))

    def can_update_job_state(self, job_id: str, job_action: JobAction) -> bool:
        """Check if an update is possible based on job state."""
        agent_ids = self._get_job_agent_ids(job_id)
        if agent_ids is None and job_action == JobAction.START:
            return True

        for agent_id in agent_ids:
            job_state = self.job_status[agent_id][job_id].state
            if job_action in JOB_ALLOWED_ACTIONS.get(job_state):
                return True

        return False

    def _get_job_agent_ids(self, job_id) -> list[str] | None:
        """Return agent_ids for job_id or None."""
        agent_ids = []
        for agent_id, jobs in self.job_status.items():
            if job_id in jobs:
                agent_ids.append(agent_id)

        if len(agent_ids):
            return agent_ids

        return None
