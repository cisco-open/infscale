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

from __future__ import annotations
from enum import Enum
from fastapi import HTTPException, status

from infscale import get_logger
from infscale.actor.job_msg import JobStatus, WorkerStatus
from infscale.config import JobConfig
from infscale.controller.ctrl_dtype import JobAction, JobActionModel

logger = None


class InvalidJobStateAction(Exception):
    """
    Custom exception for invalid actions in a job state.
    """

    def __init__(self, job_id, action, state):
        self.job_id = job_id
        self.action = action
        self.state = state

        super().__init__(
            f"Job {job_id}: '{action}' action is not allowed in the '{state}' state."
        )


class JobStateEnum(Enum):
    """JobState enum."""

    READY = "ready"
    RUNNING = "running"
    STARTING = "starting"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UPDATING = "updating"
    COMPLETE = "complete"


class BaseJobState:
    """Abstract base class for job states."""

    def __init__(self, context: JobContext):
        self.context = context
        self.job_id = context.job_id

    async def start(self):
        """Transition to STARTING state."""
        raise InvalidJobStateAction(self.job_id, "start", self.context.state_enum.value)

    async def stop(self):
        """Transition to STOPPING state."""
        raise InvalidJobStateAction(self.job_id, "stop", self.context.state_enum.value)

    async def update(self):
        """Transition to UPDATING state."""
        raise InvalidJobStateAction(
            self.job_id, "update", self.context.state_enum.value
        )

    def cond_running(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "running", self.context.state_enum.value
        )

    def cond_updated(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "updating", self.context.state_enum.value
        )

    def cond_stopped(self):
        """Handle the transition to stopped."""
        raise InvalidJobStateAction(
            self.job_id, "stopping", self.context.state_enum.value
        )

    def cond_complete(self):
        """Handle the transition to complete."""
        raise InvalidJobStateAction(
            self.job_id, "complete", self.context.state_enum.value
        )


class ReadyState(BaseJobState):
    async def start(self):
        """Transition to STARTING state."""
        agent_id = self.context._get_ctrl_agent_id()
        req = self.context.req

        self.context.set_agent_ids([agent_id])
        self.context.process_cfg(req.config)

        await self.context.prepare_config(agent_id, self.job_id, req)

        self.context.set_state(JobStateEnum.STARTING)


class RunningState(BaseJobState):
    """RunningState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        agent_id = self.context._get_ctx_agent_id()
        await self.context.ctrl._send_action_to_agent(
            agent_id, self.job_id, self.context.req
        )

        self.context.set_state(JobStateEnum.STOPPING)

    async def update(self):
        """Transition to UPDATING state."""
        agent_id = self.context._get_ctx_agent_id()

        self.context.process_cfg(self.context.req.config)
        await self.context.prepare_config(agent_id, self.job_id, self.context.req)

        self.context.set_state(JobStateEnum.UPDATING)

    def cond_complete(self):
        """Handle the transition to complete."""
        self.context.set_state(JobStateEnum.COMPLETE)


class StartingState(BaseJobState):
    """StartingState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        agent_id = self.context._get_ctx_agent_id()
        await self.context.ctrl._send_action_to_agent(
            agent_id, self.job_id, self.context.req
        )

        self.context.set_state(JobStateEnum.STOPPING)

    def cond_running(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class StoppedState(BaseJobState):
    """StoppedState class."""

    async def start(self):
        """Transition to STARTING state."""
        agent_id = self.context._get_ctrl_agent_id()
        req = self.context.req

        self.context.set_agent_ids([agent_id])
        self.context.process_cfg(req.config)

        await self.context.prepare_config(agent_id, self.job_id, req)

        self.context.set_state(JobStateEnum.STARTING)


class StoppingState(BaseJobState):
    """StoppingState class."""

    def cond_stopped(self):
        """Handle the transition to stopped."""
        self.context.set_state(JobStateEnum.STOPPED)


class UpdatingState(BaseJobState):
    """StoppingState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        agent_id = self.context._get_ctx_agent_id()
        await self.context.ctrl._send_action_to_agent(
            agent_id, self.job_id, self.context.req
        )

        self.context.set_state(JobStateEnum.STOPPING)

    def cond_updated(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class CompleteState(BaseJobState):
    """CompleteState class."""

    async def start(self):
        """Transition to STARTING state."""
        agent_id = self.context._get_ctrl_agent_id()
        req = self.context.req

        self.context.set_agent_ids([agent_id])
        self.context.process_cfg(req.config)

        await self.context.prepare_config(agent_id, self.job_id, req)

        self.context.set_state(JobStateEnum.STARTING)


class JobContext:
    """JobContext class."""

    def __init__(self, ctrl, job_id: str):
        self.ctrl = ctrl
        self.job_id = job_id
        self.state = ReadyState(self)
        self.state_enum = JobStateEnum.READY
        self.agent_ids: dict[str, JobStatus | None] = {}
        self.req: JobActionModel = None
        self.config = None
        self.new_config = None
        self.ports = None
        self.num_new_workers = 0
        self.wrk_status: dict[str, WorkerStatus] = {}

        global logger
        logger = get_logger()

    def set_agent_ids(self, agent_ids: list[str]) -> None:
        """Set a list of agents."""
        for id in agent_ids:
            self.agent_ids[id] = None

    def _set_job_status_on_agent(self, agent_id: str, job_status: JobStatus) -> None:
        """Set job status on agent id."""
        self.agent_ids[agent_id] = job_status

    def set_ports(self, ports: list[int]) -> None:
        """Set port numbers for workers."""
        self.ports = ports

    def set_new_workers_num(self, wrk_count: int) -> None:
        """Set new number of workers."""
        self.num_new_workers = wrk_count

    def set_state(self, state_enum):
        """Transition the job to a new state."""
        self.state_enum = state_enum
        self.state = self._get_state_class(state_enum)(self)
        logger.info(f"current state for {self.job_id} is {self.state_enum}")

    def handle_job_status(self, status: str, agent_id: str) -> None:
        """Handle job status received from the agent."""
        try:
            status_enum = JobStatus(status)
            self.agent_ids[agent_id] = status_enum
            self._do_cond(status_enum)
        except InvalidJobStateAction as e:
            logger.warning(e)
        except ValueError:
            logger.warning(f"'{status}' is not a valid JobStatus")

    def _do_cond(self, status: JobStatus) -> None:
        """Handle job status by calling conditional action."""
        match status:
            case JobStatus.RUNNING:
                self.cond_running()
            case JobStatus.COMPLETED:
                self.cond_complete()
            case JobStatus.STOPPED:
                self.cond_stopped()
            case JobStatus.UPDATED:
                self.cond_updated()
            case _:
                logger.warning(f"unsupported job status: '{status}'")

    def set_wrk_status(self, wrk_id, status: str) -> None:
        """Set worker status."""
        try:
            status_enum = WorkerStatus(status)
            self.wrk_status[wrk_id] = status_enum
        except ValueError:
            logger.warning(f"'{status}' is not a valid WorkerStatus")

    def process_cfg(self, new_cfg: JobConfig) -> None:
        """Process received config from controller."""
        if new_cfg is None:
            return
        self.num_new_workers = self._get_new_workers_count(self.config, new_cfg)
        self.new_config = new_cfg

    async def prepare_config(
        self, agent_id: str, job_id: str, req: JobActionModel
    ) -> None:
        # fetch port numbers from agent
        await self.ctrl._job_setup(agent_id, req.config)

        # update job config
        await self.ctrl._patch_job_cfg(agent_id, job_id)

        # schedule config transfer to agent after job setup is done
        await self.ctrl._send_config_to_agent(agent_id, job_id, req)

    def _get_new_workers_count(self, config: JobConfig, new_cfg: JobConfig) -> int:
        """Return the number of new workers between and old and new config."""
        curr_workers = []
        if config is not None:
            curr_workers = [
                w.name for w_list in config.flow_graph.values() for w in w_list
            ]
        new_workers = [w.name for w_list in new_cfg.flow_graph.values() for w in w_list]

        return len(set(new_workers) - set(curr_workers))

    def _get_state_class(self, state_enum):
        """Map a JobStateEnum to its corresponding state class."""
        state_mapping = {
            JobStateEnum.READY: ReadyState,
            JobStateEnum.RUNNING: RunningState,
            JobStateEnum.STARTING: StartingState,
            JobStateEnum.STOPPED: StoppedState,
            JobStateEnum.STOPPING: StoppingState,
            JobStateEnum.UPDATING: UpdatingState,
            JobStateEnum.COMPLETE: CompleteState,
        }
        return state_mapping[state_enum]

    def _get_ctrl_agent_id(self) -> list[str] | None:
        agent_ids = list(self.ctrl.agent_contexts.keys())

        self._check_agent_ids(agent_ids)

        # TODO: add support for multiple agent ids
        return agent_ids[0]

    def _get_ctx_agent_id(self) -> list[str] | None:
        """Return current agent id."""
        self._check_agent_ids(self.agent_ids)

        agent_id = next(iter(self.agent_ids))
        # TODO: add support for multiple agent ids
        return agent_id

    def _check_agent_ids(self, agent_ids: list[str]) -> None:
        """Check available agent ids or raise exception."""
        if len(agent_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No agent found",
            )

    async def do(self, req: JobActionModel):
        """Handle specific action"""
        self.req = req

        match req.action:
            case JobAction.START:
                await self.start()

            case JobAction.UPDATE:
                await self.update()

            case JobAction.STOP:
                await self.stop()
            case _:
                raise InvalidJobStateAction(
                    self.job_id, req.action, self.state_enum.value
                )

    def _check_job_status_on_all_agents(self, job_status: JobStatus) -> bool:
        """Return true or false if all agents have the same job status."""
        return all(status == job_status for status in self.agent_ids.values())

    async def start(self):
        """Transition to STARTING state."""
        await self.state.start()

    async def stop(self):
        """Transition to STOPPING state."""
        await self.state.stop()

    async def update(self):
        """Transition to UPDATING state."""
        await self.state.update()

    def cond_running(self):
        """Handle the transition to running."""
        all_agents_running = self._check_job_status_on_all_agents(JobStatus.RUNNING)

        if all_agents_running:
            self.state.cond_running()

    def cond_updated(self):
        """Handle the transition to running."""
        # TODO: handle updated condition later
        all_agents_running = self._check_job_status_on_all_agents(JobStatus.UPDATED)

        if all_agents_running:
            self.state.cond_updated()

    def cond_stopped(self):
        """Handle the transition to stopped."""
        all_agents_stopped = self._check_job_status_on_all_agents(JobStatus.STOPPED)

        if all_agents_stopped:
            self.state.cond_stopped()

    def cond_complete(self):
        """Handle the transition to complete."""
        all_agents_completed = self._check_job_status_on_all_agents(JobStatus.COMPLETED)

        if all_agents_completed:
            self.state.cond_complete()
