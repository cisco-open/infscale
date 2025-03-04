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

"""job_context.py."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Iterator

from fastapi import HTTPException, status
from infscale import get_logger
from infscale.actor.job_msg import JobStatus, WorkerStatus
from infscale.config import JobConfig, WorkerData, WorldInfo
from infscale.controller.ctrl_dtype import CommandAction, CommandActionModel
from infscale.controller.exceptions import (InfScaleException,
                                            InvalidJobStateAction)

if TYPE_CHECKING:
    from infscale.controller.controller import Controller

logger = None


class AgentMetaData:
    """AgentMetaData class."""

    def __init__(
        self,
        id: str = None,
        ip: str = None,
        job_status: JobStatus = None,
        config: JobConfig = None,
        new_config: JobConfig = None,
        num_new_worlds: int = 0,
        ports: list[int] = None,
    ):
        """Initialize AgentMedataData instance."""
        self.id = id
        self.ip = ip
        self.job_status = job_status
        self.config = config
        self.new_config = new_config
        self.num_new_worlds = num_new_worlds
        self.ports = ports
        self.job_setup_event = asyncio.Event()
        self.ready_to_config = False
        self.wids_to_deploy: list[str] = []
        self.existing_workers: set[str] = set()


class JobStateEnum(Enum):
    """JobState enum."""

    READY = "ready"
    RUNNING = "running"
    STARTING = "starting"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UPDATING = "updating"
    COMPLETING = "completing"
    COMPLETE = "complete"
    FAILED = "failed"


class BaseJobState:
    """Abstract base class for job states."""

    def __init__(self, context: JobContext):
        """Initialize BaseJobState instance."""
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

    async def cond_completing(self):
        """Handle the transition to completing."""
        raise InvalidJobStateAction(
            self.job_id, "completing", self.context.state_enum.value
        )

    def cond_complete(self):
        """Handle the transition to complete."""
        raise InvalidJobStateAction(
            self.job_id, "complete", self.context.state_enum.value
        )


class ReadyState(BaseJobState):
    """ReadyState class."""

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)


class RunningState(BaseJobState):
    """RunningState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context.send_command_to_agents(self.context.req)
        self.context.set_state(JobStateEnum.STOPPING)

    async def update(self):
        """Transition to UPDATING state."""
        agent_ids = self.context._get_ctx_agent_ids()

        self.context.process_cfg(agent_ids)

        tasks = []

        for info in self.context.running_agent_info:
            task = asyncio.create_task(self.context.prepare_config(info.id))
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.context.set_state(JobStateEnum.UPDATING)

    async def cond_completing(self):
        """Handle the transition to completing."""
        server_wids = self.context.get_server_wids()

        verdict = all(
            self.context.get_wrk_status(wid) == WorkerStatus.DONE for wid in server_wids
        )

        if not verdict:
            return

        command = CommandActionModel(
            action=CommandAction.FINISH_JOB, job_id=self.job_id
        )
        await self.context.send_command_to_agents(command)
        self.context.set_state(JobStateEnum.COMPLETING)


class StartingState(BaseJobState):
    """StartingState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context.send_command_to_agents(self.context.req)
        self.context.set_state(JobStateEnum.STOPPING)

    def cond_running(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class StoppedState(BaseJobState):
    """StoppedState class."""

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)


class StoppingState(BaseJobState):
    """StoppingState class."""

    def cond_stopped(self):
        """Handle the transition to stopped."""
        self.context.set_state(JobStateEnum.STOPPED)


class CompletingState(BaseJobState):
    """CompletingState class."""

    async def cond_completing(self):
        """Handle the transition to completing.

        This is executed because non-server workers send DONE status message
        once server workers are DONE. In this case, we don't need to do
        anything. So, we simply return here.
        """
        return

    def cond_complete(self):
        """Handle the transition to complete."""
        self.context.set_state(JobStateEnum.COMPLETE)


class UpdatingState(BaseJobState):
    """StoppingState class."""

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context.send_command_to_agents(self.context.req)
        self.context.set_state(JobStateEnum.STOPPING)

    def cond_updated(self):
        """Handle the transition to running."""
        self.context.set_state(JobStateEnum.RUNNING)


class CompleteState(BaseJobState):
    """CompleteState class."""

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)

    async def cond_completing(self):
        """Handle the transition to completing.

        This is executed because non-server workers send DONE status message
        once server workers are DONE. In this case, we don't need to do
        anything. So, we simply return here.
        """
        return


class FailedState(BaseJobState):
    """FailedState class."""

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)


class JobContext:
    """JobContext class."""

    def __init__(self, ctrl: Controller, job_id: str):
        """Initialize JobContext instance."""
        self.ctrl = ctrl
        self.job_id = job_id
        self.state = ReadyState(self)
        self.state_enum = JobStateEnum.READY
        self.agent_info: dict[str, AgentMetaData] = {}
        self.req: CommandActionModel = None
        self.wrk_status: dict[str, WorkerStatus] = {}
        # event to update the config after all agents added ports and ip address
        self.agents_setup_event = asyncio.Event()
        # list of agent ids that will deploy workers
        self.running_agent_info: list[AgentMetaData] = []

        global logger
        logger = get_logger()

    def get_agent_data(self, agent_id: str) -> AgentMetaData:
        """Return agent metadata."""
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

    async def send_command_to_agents(self, command: CommandActionModel) -> None:
        """Send command to all agents in the job."""
        tasks = []

        agent_ids = self._get_ctx_agent_ids()
        for aid in agent_ids:
            task = self.ctrl._send_command_to_agent(aid, self.job_id, command)
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def do_wrk_cond(self, wid: str, status: WorkerStatus) -> None:
        """Handle worker status by calling conditional action."""
        match status:
            case WorkerStatus.DONE:
                await self.cond_completing()

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

    def get_wrk_status(self, wrk_id: str) -> WorkerStatus:
        """Get worker status."""
        return self.wrk_status[wrk_id]

    def set_wrk_status(self, wrk_id: str, status: WorkerStatus) -> None:
        """Set worker status."""
        self.wrk_status[wrk_id] = status

    def process_cfg(self, agent_ids: list[str]) -> None:
        """Process received config from controller and set a deployer of agent ids."""
        agent_data = self._get_agents_data(agent_ids)

        agent_cfg, wrk_distribution = self.ctrl.deploy_policy.split(
            agent_data, self.req.config
        )

        self._update_agent_data(agent_cfg, wrk_distribution)

        # create a list of agent info that will deploy workers
        running_agent_info = [
            self.agent_info[agent_id] for agent_id in wrk_distribution.keys()
        ]

        self.running_agent_info = running_agent_info

    def _get_agents_data(self, agent_ids: list[str]) -> list[AgentMetaData]:
        """Get a list of agent metadata given agent ids."""
        result = []

        for agent_id in agent_ids:
            data = self.agent_info[agent_id]
            result.append(data)

        return result

    def _update_agent_data(
        self, agent_cfg: dict[str, JobConfig], wrk_distribution: dict[str, set[str]]
    ) -> None:
        """Update agent data based on deployment policy split."""
        for agent_id, new_cfg in agent_cfg.items():
            agent_data = self.agent_info[agent_id]
            agent_data.new_config = new_cfg
            agent_data.num_new_worlds = self._count_new_worlds(
                agent_data.config, new_cfg
            )
            agent_data.wids_to_deploy = self._get_deploy_worker_ids(new_cfg.workers)
            agent_data.existing_workers = wrk_distribution[agent_id]

    async def prepare_config(self, agent_id: str) -> None:
        """Prepare config for deploy."""
        agent_data = self.agent_info[agent_id]
        # fetch port numbers from agent
        await self.ctrl._job_setup(agent_data)

        await agent_data.job_setup_event.wait()

        # agent is ready to perform setup
        agent_data.ready_to_config = True
        if any(info.ready_to_config is False for info in self.running_agent_info):
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
        config, new_config = agent_data.config, agent_data.new_config

        curr_worlds: dict[str, WorldInfo] = {}
        if config is not None:
            for world_list in config.flow_graph.values():
                for world in world_list:
                    curr_worlds[world.name] = world

        world_agent_map = self._get_world_agent_map(new_config)
        agent_port_map = self._get_agent_port_map()

        # step 2: patch new config with existing world ports and assign ports to new ones
        for world_list in new_config.flow_graph.values():
            for world in world_list:
                if world.name in curr_worlds:
                    # keep existing ports
                    world.addr = curr_worlds[world.name].addr
                    world.data_port = curr_worlds[world.name].data_port
                    world.ctrl_port = curr_worlds[world.name].ctrl_port
                else:
                    agent_info = world_agent_map[world.name]
                    port_iter = agent_port_map[agent_info.id]
                    addr = self.ctrl.agent_contexts[agent_info.id].ip

                    # assign new ports to new worlds
                    world.addr = addr
                    world.data_port = next(port_iter)
                    world.ctrl_port = next(port_iter)

        agent_data.config = new_config
        agent_data.new_config = None
        agent_data.num_new_worlds = 0

        agent_data.job_setup_event.clear()

    def _get_agent_port_map(self) -> dict[str, Iterator[int]]:
        """Create map between agent id and available ports."""
        agent_ports = {}

        for info in self.running_agent_info:
            agent_ports[info.id] = iter(info.ports)

        return agent_ports

    def _get_world_agent_map(self, config: JobConfig) -> dict[str, AgentMetaData]:
        """Create map between world name and agent."""
        world_agent_map = {}

        for wid, world_list in config.flow_graph.items():
            agent_for_deploy = self._get_agent_by_worker_id(wid)
            for world in world_list:
                world_agent_map[world.name] = agent_for_deploy

        return world_agent_map

    def _get_agent_by_worker_id(self, wid: str) -> AgentMetaData:
        """Return agent that will deploy a given worker id."""
        return next(
            (info for info in self.agent_info.values() if wid in info.wids_to_deploy),
            None,
        )

    def _count_new_worlds(self, config: JobConfig, new_cfg: JobConfig) -> int:
        """Return the number of new worlds between and old and new config."""
        curr_worlds = []
        if config is not None:
            curr_worlds = self._get_world_names_to_setup(config)

        new_worlds = self._get_world_names_to_setup(new_cfg)
        return len(set(new_worlds) - set(curr_worlds))

    def _get_world_names_to_setup(self, config: JobConfig) -> list[str]:
        """Return a list of world names to be set up."""
        worker_ids = self._get_deploy_worker_ids(config.workers)
        world_names = [
            world.name
            for wid, world_list in config.flow_graph.items()
            for world in world_list
            if wid in worker_ids
        ]

        return world_names

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
            JobStateEnum.COMPLETING: CompletingState,
            JobStateEnum.COMPLETE: CompleteState,
            JobStateEnum.FAILED: FailedState,
        }
        return state_mapping[state_enum]

    def _manage_agent_metadata(self, agent_ids, agent_ips: list[str]) -> None:
        """Manage agent metadata by create/update/delete."""
        # create or update AgentMetaData
        for id, ip in zip(agent_ids, agent_ips):
            if id not in self.agent_info:
                self.agent_info[id] = AgentMetaData(id=id, ip=ip)
            else:
                self.agent_info[id].ip = ip

        s = set(agent_ids)
        for id in list(self.agent_info.keys()):
            if id in s:
                continue

            del self.agent_info[id]

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

    def get_server_wids(self) -> list[str]:
        """Return a list of worker ids whose role is a server."""
        config = next(iter(self.agent_info.values())).config
        wids = set()

        for worker in config.workers:
            if not worker.is_server:
                continue
            wids.add(worker.id)

        return list(wids)

    async def do(self, req: CommandActionModel):
        """Handle specific action."""
        self.req = req

        match req.action:
            case CommandAction.START:
                await self.start()

            case CommandAction.UPDATE:
                await self.update()

            case CommandAction.STOP:
                await self.stop()
            case _:
                raise InvalidJobStateAction(
                    self.job_id, req.action, self.state_enum.value
                )

    def _check_job_status_on_all_agents(self, job_status: JobStatus) -> bool:
        """Return true or false if all agents have the same job status."""
        return all(info.job_status == job_status for info in self.running_agent_info)

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

    async def cond_completing(self):
        """Handle the transition to completing."""
        await self.state.cond_completing()

    async def __start(self):
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., ReadyState, CompleteState, etc).
        agent_ids = list(self.ctrl.agent_contexts.keys())
        agent_ips = [ctx.ip for ctx in self.ctrl.agent_contexts.values()]

        self._check_agent_ids(agent_ids)

        self._manage_agent_metadata(agent_ids, agent_ips)

        self.process_cfg(agent_ids)

        tasks = []

        for info in self.running_agent_info:
            task = asyncio.create_task(self.prepare_config(info.id))
            tasks.append(task)

        await asyncio.gather(*tasks)
