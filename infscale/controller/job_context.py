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
import copy
from enum import Enum
from typing import TYPE_CHECKING, Iterator

from fastapi import HTTPException, status

from infscale import get_logger
from infscale.common.exceptions import (
    InfScaleException,
    InsufficientResources,
    InvalidConfig,
    InvalidJobStateAction,
)
from infscale.common.job_msg import WorkerStatus
from infscale.common.metrics import PerfMetrics
from infscale.configs.job import JobConfig, WorldInfo
from infscale.controller.agent_context import (
    CPU_LOAD_THRESHOLD,
    AgentResources,
    DeviceType,
)
from infscale.controller.ctrl_dtype import CommandAction, CommandActionModel
from infscale.controller.deployment.assignment import AssignmentCollection
from infscale.controller.job_checker import JobChecker


if TYPE_CHECKING:
    from infscale.controller.controller import Controller


logger = None


class AgentMetaData:
    """AgentMetaData class."""

    def __init__(
        self,
        id: str = None,
        ip: str = None,
        num_new_worlds: int = 0,
        ports: list[int] = None,
    ):
        """Initialize AgentMedataData instance."""
        self.id = id
        self.ip = ip
        self.num_new_worlds = num_new_worlds
        self.ports = ports
        self.job_setup_event = asyncio.Event()
        self.ready_to_config = False
        self.wids_to_deploy: set[str] = set()
        self.assignment_coll = AssignmentCollection()
        self.past_assignment_coll = AssignmentCollection()


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
    FAILING = "failing"
    RECOVERY = "recovery"


class BaseJobState:
    """Abstract base class for job states."""

    def __init__(self, context: JobContext):
        """Initialize BaseJobState instance."""
        self.context = context
        self.job_id = context.job_id

    def on_exit(self) -> None:
        """Cleanup when the state gets destroyed."""
        pass

    def enum_(self) -> JobStateEnum:
        """Return the state enum."""
        pass

    async def start(self):
        """Transition to STARTING state."""
        raise InvalidJobStateAction(
            self.job_id, "start", self.context.state.enum_().value
        )

    async def stop(self):
        """Transition to STOPPING state."""
        raise InvalidJobStateAction(
            self.job_id, "stop", self.context.state.enum_().value
        )

    async def update(self):
        """Transition to UPDATING state."""
        raise InvalidJobStateAction(
            self.job_id, "update", self.context.state.enum_().value
        )

    def cond_running(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "running", self.context.state.enum_().value
        )

    def cond_updated(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "updating", self.context.state.enum_().value
        )

    def cond_stopped(self):
        """Handle the transition to stopped."""
        raise InvalidJobStateAction(
            self.job_id, "stopping", self.context.state.enum_().value
        )

    async def cond_completing(self):
        """Handle the transition to completing."""
        raise InvalidJobStateAction(
            self.job_id, "completing", self.context.state.enum_().value
        )

    def cond_complete(self):
        """Handle the transition to complete."""
        raise InvalidJobStateAction(
            self.job_id, "complete", self.context.state.enum_().value
        )

    async def cond_failing(self):
        """Handle the transition to failing."""
        raise InvalidJobStateAction(
            self.job_id, "failing", self.context.state.enum_().value
        )

    async def cond_recovery(self):
        """Handle the transition to recovery."""
        raise InvalidJobStateAction(
            self.job_id, "recovery", self.context.state.enum_().value
        )


class ReadyState(BaseJobState):
    """ReadyState class."""

    def enum_(self) -> JobStateEnum:
        """Return ready state enum."""
        return JobStateEnum.READY

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

    def enum_(self) -> JobStateEnum:
        """Return running state enum."""
        return JobStateEnum.RUNNING

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context._JobContext__stop()

    async def update(self):
        """Transition to UPDATING state."""
        await self.context._JobContext__update()
        self.context.set_state(JobStateEnum.UPDATING)

    async def cond_completing(self):
        """Handle the transition to completing."""
        await self.context._JobContext__cond_completing()

    async def cond_recovery(self):
        """Handle the transition to recovery."""
        self.context.set_state(JobStateEnum.RECOVERY)


class StartingState(BaseJobState):
    """StartingState class."""

    def enum_(self) -> JobStateEnum:
        """Return starting state enum."""
        return JobStateEnum.STARTING

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context._JobContext__stop()

    def cond_running(self):
        """Handle the transition to running."""
        if self.context.in_statuses_for_all_workers({WorkerStatus.RUNNING}):
            self.context.set_state(JobStateEnum.RUNNING)


class StoppedState(BaseJobState):
    """StoppedState class."""

    def __init__(self, context: JobContext):
        """Initialize StoppedState instance."""
        super().__init__(context)
        self.context.cleanup()

    def enum_(self) -> JobStateEnum:
        """Return stopped state enum."""
        return JobStateEnum.STOPPED

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

    def enum_(self) -> JobStateEnum:
        """Return stopping state enum."""
        return JobStateEnum.STOPPING

    def cond_stopped(self):
        """Handle the transition to stopped."""
        if self.context.in_statuses_for_all_workers({WorkerStatus.TERMINATED}):
            self.context.set_state(JobStateEnum.STOPPED)


class FailingState(BaseJobState):
    """FailingState class."""

    def enum_(self) -> JobStateEnum:
        """Return failing state enum."""
        return JobStateEnum.FAILING

    def cond_stopped(self):
        """Handle the transition to failed."""
        if self.context.in_statuses_for_all_workers(
            {WorkerStatus.TERMINATED, WorkerStatus.FAILED}
        ):
            self.context.set_state(JobStateEnum.FAILED)


class CompletingState(BaseJobState):
    """CompletingState class."""

    def enum_(self) -> JobStateEnum:
        """Return completing state enum."""
        return JobStateEnum.COMPLETING

    def cond_complete(self):
        """Handle the transition to complete."""
        if self.context.in_statuses_for_all_workers({WorkerStatus.DONE}):
            self.context.set_state(JobStateEnum.COMPLETE)


class UpdatingState(BaseJobState):
    """UpdatingState class."""

    def enum_(self) -> JobStateEnum:
        """Return updating state enum."""
        return JobStateEnum.UPDATING

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context._JobContext__stop()

    def cond_running(self):
        """Handle the transition to running."""
        statuses = {WorkerStatus.RUNNING, WorkerStatus.UPDATED}
        if self.context.in_statuses_for_all_workers(statuses):
            self.context.set_state(JobStateEnum.RUNNING)

    def cond_updated(self):
        """Handle the transition to running."""
        statuses = {WorkerStatus.RUNNING, WorkerStatus.UPDATED}
        if self.context.in_statuses_for_all_workers(statuses):
            self.context.set_state(JobStateEnum.RUNNING)

    async def cond_completing(self):
        """Handle the transition to completing."""
        await self.context._JobContext__cond_completing()

    async def cond_failing(self):
        """Handle the transition to failing."""
        await self.context._JobContext__cond_failing()


class CompleteState(BaseJobState):
    """CompleteState class."""

    def __init__(self, context: JobContext):
        """Initialize CompleteState instance."""
        super().__init__(context)
        self.context.cleanup()

    def enum_(self) -> JobStateEnum:
        """Return complete state enum."""
        return JobStateEnum.COMPLETE

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)


class FailedState(BaseJobState):
    """FailedState class."""

    def __init__(self, context: JobContext):
        """Initialize FailedState instance."""
        super().__init__(context)
        self.context.cleanup()

    def enum_(self) -> JobStateEnum:
        """Return failed state enum."""
        return JobStateEnum.FAILED

    async def start(self):
        """Transition to STARTING state."""
        try:
            await self.context._JobContext__start()
        except InfScaleException as e:
            self.context.set_state(JobStateEnum.FAILED)
            raise e

        self.context.set_state(JobStateEnum.STARTING)


class RecoveryState(BaseJobState):
    """RecoveryState class."""

    def __init__(self, context: JobContext):
        """Initialize RecoveryState instance."""
        super().__init__(context)
        self.recovery_task = asyncio.create_task(self._start_recovery())

    def on_exit(self) -> None:
        """Cleanup when the state gets destroyed."""
        self.recovery_task.cancel()

    async def _start_recovery(self) -> None:
        """Start recovery tasks for failed workers."""
        failed_wrk_ids = {
            k for k, v in self.context.wrk_status.items() if v == WorkerStatus.FAILED
        }

        while True:
            wrk_resources_map = self._get_wrk_resources_map(failed_wrk_ids)
            if len(wrk_resources_map) == len(failed_wrk_ids):
                break

            await asyncio.sleep(1)

        cfg = self.context.get_recovery_updated_config(wrk_resources_map)
        self.context.req.config = cfg

        await self.context._JobContext__update()

    def cond_updated(self):
        """Handle the transition to running."""
        statuses = {WorkerStatus.RUNNING, WorkerStatus.UPDATED}
        if self.context.in_statuses_for_all_workers(statuses):
            self.context.set_state(JobStateEnum.RUNNING)

    def cond_running(self):
        """Handle the transition to running."""
        statuses = {WorkerStatus.RUNNING, WorkerStatus.UPDATED}
        if self.context.in_statuses_for_all_workers(statuses):
            self.context.set_state(JobStateEnum.RUNNING)

    def _get_wrk_resources_map(self, wrk_ids: set[str]) -> dict[str, str]:
        """Create map between worker ID and agent resources."""
        agent_resources = self.context.get_agent_resources_map()
        wrk_agent_map: dict[str, tuple[str, int]] = {}
        agent_gpu_map: dict[str, set[int]] = {}

        for wrk_id in wrk_ids:
            curr_agent = self._get_curr_agent_data(wrk_id)
            assign_success = self._assign_available_gpu_to_worker(
                curr_agent.id, agent_resources[curr_agent.id], wrk_id, wrk_agent_map, agent_gpu_map
            )

            if not assign_success:
                assign_success = self._search_gpu_on_all_agents(
                    agent_resources, curr_agent.id, wrk_id, wrk_agent_map, agent_gpu_map
                )

            if not assign_success:
                # if no resources, return and let while loop continue
                return {}

        return wrk_agent_map

    def _get_curr_agent_data(self, wrk_id: str) -> AgentMetaData:
        """Return current agent that deployed worker ID."""
        agent_data = next(
            agent_data
            for agent_data in self.context.running_agent_info.values()
            if wrk_id in agent_data.wids_to_deploy
        )

        return agent_data

    def _add_gpu_to_agent(self, agent_id: str, gpu_id: int, agent_gpu_map: dict[str, set[int]]) -> None:
        """Add a GPU ID to the list of GPUs associated with the given agent."""
        if agent_id not in agent_gpu_map:
            agent_gpu_map[agent_id] = set()

        agent_gpu_map[agent_id].add(gpu_id)

    def _assign_available_gpu_to_worker(
        self,
        agent_id: str,
        resources: AgentResources,
        wrk_id: str,
        wrk_agent_map: dict[str, tuple[str, int]],
        agent_gpu_map: dict[str, set[int]]
    ) -> bool:
        """Attempt to assign an available GPU from the given agent to the worker.

        Updates internal mappings if successful.

        Returns:
            bool: True if a GPU was successfully assigned, False otherwise.
        """
        for gpu in resources.gpu_stats:
            if not gpu.used and gpu.id not in agent_gpu_map.get(agent_id, []):
                self._add_gpu_to_agent(agent_id, gpu.id, agent_gpu_map)
                agent_info = self.context.agent_info[agent_id]
                wrk_agent_map[wrk_id] = (agent_info.ip, gpu.id)

                return True

        return False

    def _search_gpu_on_all_agents(
        self,
        agent_resources: dict[str, AgentResources],
        curr_agent_id: str,
        wrk_id: str,
        wrk_agent_map: dict[str, tuple[str, int]],
        agent_gpu_map: dict[str, set[int]]
    ) -> bool:
        """Look for available resources on all agents and attempt to assigned unused GPU.
        
        Returns:
            bool: True if a GPU was successfully assigned, False otherwise.
        """
        for agent_id, resources in agent_resources.items():
            if agent_id == curr_agent_id:
                continue

            return self._assign_available_gpu_to_worker(
                agent_id, resources, wrk_id, wrk_agent_map, agent_gpu_map
            )

        return False

    def enum_(self) -> JobStateEnum:
        """Return recovery state enum."""
        return JobStateEnum.RECOVERY

    async def stop(self):
        """Transition to STOPPING state."""
        await self.context._JobContext__stop()

    async def cond_completing(self):
        """Handle the transition to completing."""
        await self.context._JobContext__cond_completing()

    async def cond_failing(self):
        """Handle the transition to failing."""
        await self.context._JobContext__cond_failing()


class JobContext:
    """JobContext class."""

    def __init__(self, ctrl: Controller, job_id: str):
        """Initialize JobContext instance."""
        self.ctrl = ctrl
        self.job_id = job_id
        self.state = ReadyState(self)
        self.agent_info: dict[str, AgentMetaData] = {}
        self.req: CommandActionModel = None
        self.wrk_status: dict[str, WorkerStatus] = {}
        self.wrkr_metrics: dict[str, PerfMetrics] = {}
        self._server_ids: set[str] = set()

        self._cur_cfg = None
        self._new_cfg = None
        self._flow_graph_patched = False

        # event to update the config after all agents added ports and ip address
        self.agents_setup_event = asyncio.Event()
        # list of agent ids that will deploy workers
        self.running_agent_info: dict[str, AgentMetaData] = {}
        self.past_running_agent_info: dict[str, AgentMetaData] = {}
        self.job_checker = JobChecker(self.wrk_status)

        self._desired_rate = 0.0

        global logger
        logger = get_logger()

    def set_desired_rate(self, rate: float) -> None:
        """Set diresed output rate for a job."""
        self._desired_rate = rate

    def get_agent_data(self, agent_id: str) -> AgentMetaData:
        """Return agent metadata."""
        return self.agent_info[agent_id]

    def set_ports(self, agent_id: str, ports: list[int]) -> None:
        """Set port numbers for workers."""
        agent_data = self.agent_info[agent_id]
        agent_data.ports = ports
        agent_data.job_setup_event.set()

    def set_state(self, state_enum: JobStateEnum):
        """Transition the job to a new state."""
        # do cleanup on current state before transitioning
        self.state.on_exit()

        self.state = self._get_state_class(state_enum)(self)
        logger.info(f"current state for {self.job_id} is {state_enum}")

    async def send_command_to_agents(self, command: CommandActionModel) -> None:
        """Send command to all agents in the job."""
        tasks = []

        for agent_id in self.agent_info.keys():
            task = self.ctrl._send_command_to_agent(agent_id, self.job_id, command)
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def do_wrk_cond(self, wid: str, status: WorkerStatus) -> None:
        """Handle worker status by calling conditional action."""
        match status:
            case WorkerStatus.RUNNING:
                self.cond_running()

            case WorkerStatus.UPDATED:
                self.cond_updated()

            case WorkerStatus.SERVING_DONE:
                await self.cond_completing()

            case WorkerStatus.DONE:
                self._release_gpu_resource_by_worker_id(wid)
                self.cond_complete()

            case WorkerStatus.FAILED:
                self._release_gpu_resource_by_worker_id(wid)
                await self.cond_recovery()

            case WorkerStatus.TERMINATED:
                self._release_gpu_resource_by_worker_id(wid)
                self.cond_stopped()

    def get_wrk_status(self, wrk_id: str) -> WorkerStatus:
        """Get worker status."""
        return self.wrk_status[wrk_id]

    def set_wrk_status(self, wrk_id: str, status: WorkerStatus) -> None:
        """Set worker status."""
        self.wrk_status[wrk_id] = status

    async def handle_agent_failure(self, agent_id: str) -> None:
        """Handle agent failure."""
        agent_data = self.agent_info.pop(agent_id, None)

        if agent_data is None:
            return

        for wid in agent_data.wids_to_deploy:
            self.set_wrk_status(wid, WorkerStatus.FAILED)

        await self.handle_potential_job_failure()

    async def handle_potential_job_failure(self) -> None:
        """Decide job failure and stop all workers."""
        job_failed = self.job_checker.is_job_failed()

        if job_failed:
            command = CommandActionModel(action=CommandAction.STOP, job_id=self.job_id)

            await self.send_command_to_agents(command)

            self.set_state(JobStateEnum.FAILED)

    def get_wrkr_metrics(self, wrkr_id: str) -> PerfMetrics:
        """Get worker's performance metrics.

        If the metrics object doesn't exist, create a new one and return it.
        """
        if wrkr_id not in self.wrkr_metrics:
            self.wrkr_metrics[wrkr_id] = PerfMetrics()

        return self.wrkr_metrics[wrkr_id]

    def set_wrkr_metrics(self, wrkr_id: str, metrics: PerfMetrics) -> None:
        """Set worker's performance metrics."""
        self.wrkr_metrics[wrkr_id] = metrics

    def _init_wrk_status(self) -> None:
        """Create worker status dict with default status."""
        for w in self._new_cfg.workers:
            if w.id not in self.wrk_status:
                self.wrk_status[w.id] = WorkerStatus.READY

    def process_cfg(self) -> None:
        """Process received config from controller and set a deployer of agent ids."""
        self._new_cfg = self.ctrl.planner.build_config(
            self.req.config,
            self.ctrl.agent_contexts,
            self._desired_rate,
            self._cur_cfg,
        )

        if JobConfig.is_identical(self._cur_cfg, self._new_cfg):
            raise InvalidConfig("current and new configs are identical")

        self._init_wrk_status()

        self._new_cfg.reqgen_config = self.ctrl.reqgen_config

        agent_resources = self.get_agent_resources_map()

        dev_type = self._decide_dev_type(agent_resources, self._new_cfg)

        assignment_map = self.ctrl.deploy_policy.split(
            dev_type, self.agent_info, agent_resources, self._new_cfg
        )

        self._update_agent_data(assignment_map)

        # create a list of agent info that will deploy workers
        running_agent_info: dict[str, AgentMetaData] = {
            agent_id: self.agent_info[agent_id] for agent_id in assignment_map.keys()
        }

        self.past_running_agent_info = self.running_agent_info
        self.running_agent_info = running_agent_info

        self.job_checker.setup(self._new_cfg)

    def get_recovery_updated_config(self, wrk_resource_map: dict[str, str]) -> JobConfig:
        """Update config with recovered worker and agent data."""
        cfg = copy.deepcopy(self._cur_cfg)

        for wrk_id, (ip, gpu_id) in wrk_resource_map.items():
            self._update_recovery_flow_graph(cfg, wrk_id, ip)
            self._update_recovery_worker_data(cfg, wrk_id, gpu_id)

        self.reset_flow_graph_patch_flag()

        return cfg

    def _update_flow_graph(self, cfg: JobConfig, recover_wid: str, ip: str) -> None:
        """Update current config's flow graph based on agent info and recovered worker id."""
        recover_flow_graph = cfg.flow_graph[recover_wid]

        for world_info in recover_flow_graph:
            world_info.addr = ip
            world_info.recover = True
            world_info.recover_count += 1
            
        for world_list in cfg.flow_graph.values():
            for world_info in world_list:
                if recover_wid in world_info.peers:
                    world_info.recover = True
                    world_info.recover_count += 1

    def _update_recovery_worker_data(self, cfg: JobConfig, wrk_id: str, gpu_id: int) -> None:
        """Update worker data for recovery."""
        worker = next(worker for worker in cfg.workers if worker.id == wrk_id)
        worker.device = f"cuda:{gpu_id}"
        worker.recover = True

    def _decide_dev_type(
        self, agent_resources: dict[str, AgentResources], config: JobConfig
    ) -> DeviceType:
        """Decide device based on available resources and number of workers."""
        num_new_workers = self._get_new_workers_count(config)
        available_gpus = 0
        for res in agent_resources.values():
            if not res.gpu_stats:
                continue

            for gpu_stat in res.gpu_stats:
                if not gpu_stat.used:
                    available_gpus += 1

            if available_gpus >= num_new_workers:
                return DeviceType.GPU

        for res in agent_resources.values():
            if not res.cpu_stats:
                continue
            if res.cpu_stats.load <= CPU_LOAD_THRESHOLD:
                return DeviceType.CPU

        raise InsufficientResources(
            f"insufficient resources to start {num_new_workers} workers."
        )

    def _get_new_workers_count(self, config: JobConfig) -> int:
        """Return number of new workers."""
        existing_worker_ids = {
            wid
            for info in self.running_agent_info.values()
            for wid in info.wids_to_deploy
        }

        new_worker_ids = [
            worker.id
            for worker in config.workers
            if worker.id not in existing_worker_ids
        ]

        return len(new_worker_ids)

    def get_agent_resources_map(self) -> dict[str, AgentResources]:
        """Return map with agent resources based on given agent ids."""
        result = {}

        for agent_id in self.agent_info.keys():
            result[agent_id] = self.ctrl.agent_contexts[agent_id].resources

        return result

    def _update_agent_data(
        self, assignment_map: dict[str, AssignmentCollection]
    ) -> None:
        """Update agent data based on deployment policy split."""
        for agent_id, new_coll in assignment_map.items():
            agent_data = self.agent_info[agent_id]
            cur_coll = agent_data.assignment_coll

            agent_data.num_new_worlds = self._count_worlds_to_setup(cur_coll, new_coll)
            agent_data.wids_to_deploy = new_coll.worker_ids()
            agent_data.past_assignment_coll = cur_coll
            agent_data.assignment_coll = new_coll

    async def prepare_config(self, agent_data: AgentMetaData) -> None:
        """Prepare config for deploy."""
        # fetch port numbers from agent
        await self.ctrl.job_setup(self.job_id, agent_data)

        await agent_data.job_setup_event.wait()

        # agent is ready to perform setup
        agent_data.ready_to_config = True
        if any(
            info.ready_to_config is False for info in self.running_agent_info.values()
        ):
            await self.agents_setup_event.wait()

        # all agents have their conn data available, release the agent setup event
        self.agents_setup_event.set()

        # update job config
        cfg = self._patch_job_cfg(agent_data)

        agent_data.ready_to_config = False

        # schedule config transfer to agent after job setup is done
        await self.ctrl.send_config_to_agent(agent_data.id, cfg, self.req)

        # block agent setup event until new config is received
        self.agents_setup_event.clear()

    def _patch_job_cfg(self, agent_data: AgentMetaData) -> JobConfig:
        """Patch a config for a specific agent."""
        self._patch_flow_graph_once()

        # since we updated flow graph and the update has been reflected
        # into _cur_cfg, now we need to create a agent-specific config
        # by deepcopying _cur_cfg. And we need to update agent-specific
        # variables (i.e., to set deploy flag and device).
        cfg = copy.deepcopy(self._cur_cfg)

        # step 2: update workers devices
        for w in cfg.workers:
            assignment_data = agent_data.assignment_coll.get_assignment_data(w.id)
            if assignment_data is None:
                log = f"not setting device for {w.id}"
                log += f" since it's not deployed in {agent_data.id}"
                logger.debug(log)
                continue

            w.device = assignment_data.device
            w.deploy = True

        agent_data.num_new_worlds = 0

        agent_data.job_setup_event.clear()

        return cfg

    def _patch_flow_graph_once(self) -> None:
        if self._flow_graph_patched:
            # a flow graph is identical across agent
            # so, we only need to patch flow_graph once
            return

        curr_worlds: dict[str, WorldInfo] = {}
        if self._cur_cfg is not None:
            for world_list in self._cur_cfg.flow_graph.values():
                for world in world_list:
                    curr_worlds[world.name] = world

        world_agent_map = self._get_world_agent_map(self._new_cfg)
        agent_port_map = self._get_agent_port_map()

        # step 1: patch new config with existing world ports and assign ports to new ones
        for wid, world_list in self._new_cfg.flow_graph.items():
            for world in world_list:
                agent_data = world_agent_map[world.name]
                port_iter = agent_port_map[agent_data.id]
                
                if world.name in curr_worlds:
                    # keep existing ports
                    world.addr = curr_worlds[world.name].addr
                    world.data_port = (
                        next(port_iter)
                        if world.recover
                        else curr_worlds[world.name].data_port
                    )
                    world.ctrl_port = (
                        next(port_iter)
                        if world.recover
                        else curr_worlds[world.name].ctrl_port
                    )
                    world.backend = curr_worlds[world.name].backend
                else:
                    assignment_coll = agent_data.assignment_coll
                    assignment_data = assignment_coll.get_assignment_data(wid)

                    addr = self.ctrl.agent_contexts[agent_data.id].ip
                    backend = assignment_data.worlds_map[world.name].backend

                    # assign new ports to new worlds
                    world.addr = addr
                    world.data_port = next(port_iter)
                    world.ctrl_port = next(port_iter)
                    world.backend = backend

        # back up new config to current config
        self._cur_cfg = self._new_cfg

        self._flow_graph_patched = True

    def reset_flow_graph_patch_flag(self) -> None:
        """Reset flow graph patch flag."""
        self._flow_graph_patched = False

    def _get_agent_port_map(self) -> dict[str, Iterator[int]]:
        """Create map between agent id and available ports."""
        agent_ports = {}

        for data in self.running_agent_info.values():
            agent_ports[data.id] = iter(data.ports)

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

    def _count_worlds_to_setup(
        self, cur_coll: AssignmentCollection, new_coll: AssignmentCollection
    ) -> int:
        """Return the number of worlds that need to be set up."""
        recover_worlds = self._get_recover_world_names(cur_coll)
        curr_worlds = self._get_world_names_to_setup(self._cur_cfg, cur_coll)
        new_worlds = self._get_world_names_to_setup(self._new_cfg, new_coll)

        return len(recover_worlds | (new_worlds - curr_worlds))
    
    def _get_recover_world_names(
        self, assignment_coll: AssignmentCollection
    ) -> set[str]:
        """Return a set of world names that need to be recovered."""
        worker_ids_to_deploy = assignment_coll.worker_ids()

        world_names = set()

        for wid, world_list in self.req.config.flow_graph.items():
            for world in world_list:
                if wid in worker_ids_to_deploy and world.recover:
                    world_names.add(world.name)

        return world_names

    def _get_world_names_to_setup(
        self, config: JobConfig, assignment_coll: AssignmentCollection
    ) -> set[str]:
        """Return a set of world names to be set up."""
        if config is None:
            # no world to set up; so, return an empty set
            return set()

        worker_ids_to_deploy = assignment_coll.worker_ids()
        world_names = {
            world.name
            for wid, world_list in config.flow_graph.items()
            for world in world_list
            if wid in worker_ids_to_deploy
        }

        return world_names

    def _get_state_class(self, state_enum: JobStateEnum):
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
            JobStateEnum.FAILING: FailingState,
            JobStateEnum.RECOVERY: RecoveryState,
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

    def _check_agent_info(self) -> None:
        """Check available agent data or raise exception."""
        if len(self.agent_info) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No agent found",
            )

    def is_server(self, wrkr_id: str) -> bool:
        """Confirm if a given worker is a server (dispatcher) or not."""
        return wrkr_id in self._server_ids

    def update_server_ids(self) -> None:
        """Update server's worker ids."""
        # reset server_ids set
        self._server_ids.clear()

        for worker in self._cur_cfg.workers:
            if not worker.is_server:
                continue
            self._server_ids.add(worker.id)

    def get_server_ids(self) -> set[str]:
        """Return a set of worker ids whose role is a server."""
        # if we already have server ids stored, return them
        if len(self._server_ids) > 0:
            return self._server_ids

        self.update_server_ids()

        return self._server_ids

    def cleanup(self) -> None:
        """Do cleanup on context resources."""
        for agent_data in self.running_agent_info.values():
            self._release_gpu_resources(agent_data)

        self.agent_info = {}
        self.req = None
        self.wrk_status = {}
        self.running_agent_info = {}
        self.past_running_agent_info = {}
        self._cur_cfg = None
        self._new_cfg = None
        self._flow_graph_patched = False

    def _release_gpu_resources(self, agent_data: AgentMetaData) -> None:
        resources = self.ctrl.agent_contexts[agent_data.id].resources
        if resources is None:
            return

        dev_set = agent_data.assignment_coll.devices()

        for gpu_stat in resources.gpu_stats:
            if f"cuda:{gpu_stat.id}" not in dev_set:
                continue

            # mark unused only for gpu used in this job
            gpu_stat.used = False

    def _release_gpu_resource_by_worker_id(self, wid: str):
        running_agent_info = set(self.running_agent_info.values())
        running_agent_info |= set(self.past_running_agent_info.values())
        for agent_data in running_agent_info:
            resources = self.ctrl.agent_contexts[agent_data.id].resources
            if resources is None:
                continue

            assignment_coll = (
                agent_data.past_assignment_coll | agent_data.assignment_coll
            )

            assignment_data = assignment_coll.get_assignment_data(wid)
            if assignment_data is None:
                continue

            if "cuda" not in assignment_data.device:
                return

            gpu_id = int(assignment_data.device.split(":")[1])
            for gpu_stat in resources.gpu_stats:
                if gpu_stat.id != gpu_id:
                    continue

                # mark unused
                gpu_stat.used = False

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
                    self.job_id, req.action, self.state.enum_().value
                )

    def in_statuses_for_all_workers(self, statuses: set[WorkerStatus]) -> bool:
        """Return true if worker is in one of given statuses."""
        return all(status in statuses for status in self.wrk_status.values())

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
        self.state.cond_running()

    def cond_updated(self):
        """Handle the transition to running."""
        self.state.cond_updated()

    def cond_stopped(self):
        """Handle the transition to stopped."""
        self.state.cond_stopped()

    def cond_complete(self):
        """Handle the transition to complete."""
        self.state.cond_complete()

    async def cond_completing(self):
        """Handle the transition to completing."""
        await self.state.cond_completing()

    async def cond_failing(self):
        """Handle the transition to failing."""
        await self.state.cond_failing()

    async def cond_recovery(self):
        """Handle the transition to recovery."""
        await self.state.cond_recovery()

    async def __update(self):
        """Transition to UPDATING state."""
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., RunningState, RecoveryState, etc).
        try:
            self.process_cfg()
        except InvalidConfig as e:
            logger.warning(f"exception: {e}")
            return

        tasks = []

        for info in self.running_agent_info.values():
            task = asyncio.create_task(self.prepare_config(info))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # after prepare_config execution is done across all running agents,
        # we reset flow_graph patch flag so that it can be checked again
        # when a new config arrives to the job.
        self.reset_flow_graph_patch_flag()

        # update server ids
        self.update_server_ids()

    async def __stop(self):
        """Transition to STOPPING state."""
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., RunningState, UpdatingState, etc).
        await self.send_command_to_agents(self.req)
        self.set_state(JobStateEnum.STOPPING)

    async def __cond_failing(self):
        """Handle the transition to failing."""
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., RunningState, RecoveryState, etc).

        job_failed = self.job_checker.is_job_failed()

        if job_failed:
            command = CommandActionModel(action=CommandAction.STOP, job_id=self.job_id)

            await self.send_command_to_agents(command)

            self.set_state(JobStateEnum.FAILING)

    async def __cond_completing(self):
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., RunningState, RecoveryState, etc).
        """Handle the transition to completing."""
        server_ids = self.get_server_ids()

        verdict = all(
            self.get_wrk_status(wid) == WorkerStatus.SERVING_DONE for wid in server_ids
        )

        if not verdict:
            return

        command = CommandActionModel(
            action=CommandAction.FINISH_JOB, job_id=self.job_id
        )
        await self.send_command_to_agents(command)
        self.set_state(JobStateEnum.COMPLETING)

    async def __start(self):
        # DO NOT call this method in job_context instance or any other places.
        # Call it only in methods of a state instance
        # (e.g., ReadyState, CompleteState, etc).
        agent_ids = list(self.ctrl.agent_contexts.keys())
        agent_ips = [ctx.ip for ctx in self.ctrl.agent_contexts.values()]

        self._manage_agent_metadata(agent_ids, agent_ips)

        self._check_agent_info()

        self.process_cfg()

        tasks = []

        for info in self.running_agent_info.values():
            task = asyncio.create_task(self.prepare_config(info))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # after prepare_config execution is done across all running agents,
        # we reset flow_graph patch flag so that it can be checked again
        # when a new config arrives to the job.
        self.reset_flow_graph_patch_flag()

        # update server ids
        self.update_server_ids()
