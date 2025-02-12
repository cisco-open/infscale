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
import asyncio
from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING, Iterator
from fastapi import HTTPException, status

from infscale import get_logger
from infscale.actor.job_msg import JobStatus, WorkerStatus
from infscale.config import JobConfig, WorkerData, WorkerInfo
from infscale.controller.ctrl_dtype import CommandAction, CommandActionModel

if TYPE_CHECKING:
    from infscale.controller.controller import Controller

logger = None


class AgentMetaData:
    """AgentMetaData class."""

    def __init__(
        self,
        id: str = None,
        job_status: JobStatus = None,
        config: JobConfig = None,
        new_config: JobConfig = None,
        num_new_workers: int = 0,
        ports: list[int] = None,
    ):
        self.id = id
        self.job_status = job_status
        self.config = config
        self.new_config = new_config
        self.num_new_workers = num_new_workers
        self.ports = ports
        self.job_setup_event = asyncio.Event()
        self.ready_to_config = False
        self.wids_to_deploy: list[str] = []


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
        req = self.context.req
        num_of_workers = len(req.config.workers)
        agent_ids = self.context._get_ctrl_agent_ids(num_of_workers)

        self.context.set_agent_ids(agent_ids)
        self.context.process_cfg(agent_ids)

        tasks = []

        for agent_id in agent_ids:
            task = asyncio.create_task(self.context.prepare_config(agent_id))
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.STARTING)


class RunningState(BaseJobState):
    """RunningState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        agent_ids = self.context._get_ctx_agent_ids()

        tasks = []

        for agent_id in agent_ids:
            task = self.context.ctrl._send_command_to_agent(
                agent_id, self.job_id, self.context.req
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.STOPPING)

    async def update(self):
        """Transition to UPDATING state."""
        agent_ids = self.context._get_ctx_agent_ids()

        self.context.process_cfg(agent_ids)

        tasks = []

        for agent_id in agent_ids:
            task = asyncio.create_task(self.context.prepare_config(agent_id))
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.UPDATING)

    def cond_complete(self):
        """Handle the transition to complete."""
        self.context.set_state(JobStateEnum.COMPLETE)


class StartingState(BaseJobState):
    """StartingState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        agent_ids = self.context._get_ctx_agent_ids()

        tasks = []

        for agent_id in agent_ids:
            task = self.context.ctrl._send_command_to_agent(
                agent_id, self.job_id, self.context.req
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.STOPPING)

    def cond_running(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class StoppedState(BaseJobState):
    """StoppedState class."""

    async def start(self):
        """Transition to STARTING state."""
        req = self.context.req
        num_of_workers = len(req.config.workers)

        agent_ids = self.context._get_ctrl_agent_ids(num_of_workers)

        self.context.set_agent_ids(agent_ids)
        self.context.process_cfg(agent_ids)

        tasks = []

        for agent_id in agent_ids:
            task = asyncio.create_task(self.context.prepare_config(agent_id))
            tasks.append(task)

        await asyncio.gather(*tasks)

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
        agent_ids = self.context._get_ctx_agent_ids()

        tasks = []

        for agent_id in agent_ids:
            task = self.context.ctrl._send_command_to_agent(
                agent_id, self.job_id, self.context.req
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.STOPPING)

    def cond_updated(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class CompleteState(BaseJobState):
    """CompleteState class."""

    async def start(self):
        """Transition to STARTING state."""
        req = self.context.req
        num_of_workers = len(req.config.workers)

        agent_ids = self.context._get_ctrl_agent_ids(num_of_workers)

        self.context.set_agent_ids(agent_ids)
        self.context.process_cfg(agent_ids)

        tasks = []

        for agent_id in agent_ids:
            task = asyncio.create_task(self.context.prepare_config(agent_id))
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.STARTING)


class JobContext:
    """JobContext class."""

    def __init__(self, ctrl: Controller, job_id: str):
        self.ctrl = ctrl
        self.job_id = job_id
        self.state = ReadyState(self)
        self.state_enum = JobStateEnum.READY
        self.agent_info: dict[str, AgentMetaData] = {}
        self.req: CommandActionModel = None
        self.wrk_status: dict[str, WorkerStatus] = {}
        # event to update the config after all agents added ports and ip address
        self.agents_setup_event = asyncio.Event()

        global logger
        logger = get_logger()

    def set_agent_ids(self, agent_ids: list[str]) -> None:
        """Set a list of agents."""
        for id in agent_ids:
            self.agent_info[id] = AgentMetaData(id=id)

    def get_agent_data(self, agent_id: str) -> AgentMetaData:
        """Return agent metadata"""
        return self.agent_info[agent_id]

    def _set_job_status_on_agent(self, agent_id: str, job_status: JobStatus) -> None:
        """Set job status on agent id."""
        self.agent_info[agent_id].job_status = job_status

    def set_ports(self, agent_id: str, ports: list[int]) -> None:
        """Set port numbers for workers."""
        agent_data = self.agent_info[agent_id]
        agent_data.ports = ports
        agent_data.job_setup_event.set()

    def set_state(self, state_enum):
        """Transition the job to a new state."""
        self.state_enum = state_enum
        self.state = self._get_state_class(state_enum)(self)
        logger.info(f"current state for {self.job_id} is {self.state_enum}")

    def handle_job_status(self, status: str, agent_id: str) -> None:
        """Handle job status received from the agent."""
        try:
            status_enum = JobStatus(status)
            self.agent_info[agent_id].job_status = status_enum
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

    def process_cfg(self, agent_ids: list[str]) -> None:
        """Process received config from controller."""
        agent_cfg = self.ctrl.deploy_policy.split(agent_ids, self.req.config)

        self._update_agent_data(agent_cfg)

    def _update_agent_data(self, agent_cfg: dict[str, JobConfig]) -> None:
        """Update agent data based on deployment policy split."""
        for agent_id, new_cfg in agent_cfg.items():
            agent_data = self.agent_info[agent_id]
            agent_data.new_config = new_cfg
            agent_data.num_new_workers = self._get_new_workers_count(
                agent_data.config, new_cfg
            )
            agent_data.wids_to_deploy = self._get_deploy_worker_ids(new_cfg.workers)

    async def prepare_config(self, agent_id: str) -> None:
        """Prepare config for deploy."""
        agent_data = self.agent_info[agent_id]

        # fetch port numbers from agent
        await self.ctrl._job_setup(agent_data)

        await agent_data.job_setup_event.wait()

        # agent is ready to perform setup
        agent_data.ready_to_config = True
        if any(info.ready_to_config == False for info in self.agent_info.values()):
            await self.agents_setup_event.wait()
    
        # all agents have their conn data available, release the agent setup event
        self.agents_setup_event.set()

        # update job config
        await self._patch_job_cfg(agent_data)

        agent_data.ready_to_config = False

        # schedule config transfer to agent after job setup is done
        await self.ctrl._send_config_to_agent(agent_data, self.req)

        # block agent setup event until new config is received
        self.agents_setup_event.clear()

    async def _patch_job_cfg(self, agent_data: AgentMetaData) -> None:
        """Patch config for updated job."""
        agent_id, config, new_config, port_iter = (
            agent_data.id,
            agent_data.config,
            agent_data.new_config,
            agent_data.ports,
        )

        curr_workers: dict[str, WorkerInfo] = {}
        if config is not None:
            for worker_list in config.flow_graph.values():
                for worker in worker_list:
                    curr_workers[worker.name] = worker

        worker_agent_map = self._get_worker_agent_map(new_config)
        agent_port_map = self._get_agent_port_map()

        # step 2: patch new config with existing workers ports and assign ports to new ones
        for worker_list in new_config.flow_graph.values():
            for worker in worker_list:
                if worker.name in curr_workers:
                    # keep existing ports
                    worker.addr = curr_workers[worker.name].addr
                    worker.data_port = curr_workers[worker.name].data_port
                    worker.ctrl_port = curr_workers[worker.name].ctrl_port
                else:
                    agent_info = worker_agent_map[worker.name]
                    port_iter = agent_port_map[agent_info.id]
                    addr = self.ctrl.agent_contexts[agent_info.id].ip
    
                    # assign new ports to new workers
                    worker.addr = addr
                    worker.data_port = next(port_iter)
                    worker.ctrl_port = next(port_iter)

        agent_data.config = new_config
        agent_data.new_config = None
        agent_data.num_new_workers = 0

        agent_data.job_setup_event.clear()

    def _get_agent_port_map(self) -> dict[str, Iterator[int]]:
        """Create map between agent id and available ports."""
        agent_ports = {}

        for info in self.agent_info.values():
            agent_ports[info.id] = iter(info.ports)

        return agent_ports

    def _get_worker_agent_map(self, config: JobConfig) -> dict[str, AgentMetaData]:
        """Create map between worker id and agent for deploy."""
        worker_agent_data = {}

        for wid, worker_list in config.flow_graph.items():
            for w in worker_list:
                agent_for_deploy = self._get_agent_by_worker_id(wid)
                worker_agent_data[w.name] = agent_for_deploy

        return worker_agent_data

    def _get_agent_by_worker_id(self, wid: str) -> AgentMetaData:
        """Return agent that will deploy given worker id."""
        return next((info for info in self.agent_info.values() if wid in info.wids_to_deploy), None)

    def _get_new_workers_count(self, config: JobConfig, new_cfg: JobConfig) -> int:
        """Return the number of new workers between and old and new config."""
        curr_workers = []
        if config is not None:
            curr_workers = self._get_deploy_worker_names(config)

        new_workers = self._get_deploy_worker_names(new_cfg)
        return len(set(new_workers) - set(curr_workers))

    def _get_deploy_worker_names(self, config: JobConfig) -> list[str]:
        """Return a list of worker names to be deployed."""
        worker_ids = self._get_deploy_worker_ids(config.workers)
        worker_names = [
            w.name
            for wid, w_list in config.flow_graph.items()
            for w in w_list
            if wid in worker_ids
        ]

        return worker_names

    def _get_deploy_worker_ids(self, workers: list[WorkerData]) -> list[str]:
        """Return a list of worker ids to be deployed."""
        return [w.id for w in workers if w.deploy]

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

    def _get_ctrl_agent_ids(self, num_of_workers: int) -> list[str]:
        """Return available agent id from controller."""
        available_agents = list(self.ctrl.agent_contexts.keys())

        if len(available_agents) > num_of_workers:
            agent_ids = list(islice(available_agents, num_of_workers))
        else:
            agent_ids = available_agents

        self._check_agent_ids(agent_ids)

        return agent_ids

    def _get_ctx_agent_ids(self) -> list[str]:
        """Return current agent ids."""
        agent_ids = list(self.agent_info.keys())
        self._check_agent_ids(agent_ids)

        return agent_ids

    def _check_agent_ids(self, agent_ids: list[str]) -> None:
        """Check available agent ids or raise exception."""
        if len(agent_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No agent found",
            )

    async def do(self, req: CommandActionModel):
        """Handle specific action"""
        self.req = req

        match req.action:
            case CommandAction.START:
                await self.ctrl.get_agents_resources()
                await self.start()

            case CommandAction.UPDATE:
                await self.ctrl.get_agents_resources()
                await self.update()

            case CommandAction.STOP:
                await self.stop()
            case _:
                raise InvalidJobStateAction(
                    self.job_id, req.action, self.state_enum.value
                )

    def _check_job_status_on_all_agents(self, job_status: JobStatus) -> bool:
        """Return true or false if all agents have the same job status."""
        return all(
            info.job_status == job_status for info in list(self.agent_info.values())
        )

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
