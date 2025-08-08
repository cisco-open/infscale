# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""cfggen.py."""

import json
from collections import defaultdict

import yaml

from infscale.common.exceptions import (
    DifferentResourceAmount,
    InsufficientResources,
    InvalidConfig,
)
from infscale.configs.job import JobConfig
from infscale.configs.plan import ExecPlan
from infscale.controller.agent_context import AgentContext


class FlowList(list):  # noqa: D101
    """Custom list class for YAML flow style representation."""

    pass


def represent_flow_list(dumper, data):  # noqa: D103 E303
    # Represent this sequence in flow style, e.g. [s-0]
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# Register the representer for FlowList
yaml.add_representer(FlowList, represent_flow_list)


class CfgGen:
    """CfgGen class."""

    def __init__(
        self,
        agent_ctxts: dict[str, AgentContext],
        source: JobConfig,
        plan_list: list[ExecPlan],
        dispatcher_device="cpu",
        base_cfg: JobConfig = None,
    ):
        """Initialize an instance."""
        self._source = source
        self._plan_list = plan_list
        self._dispatcher_device = dispatcher_device
        self._base_cfg = base_cfg

        self._agent_ctxts = agent_ctxts

        self._is_auto_regressive = self._source.is_auto_regressive()

        # key: agent id and value is AgentContext
        # sort agent context by # of unused gpus in a decreasing order
        # and then by agent id in an increasing order to break a tie
        tmp = sorted(agent_ctxts.items(), key=lambda i: (-i[1].avail_gpu_count(), i[0]))
        self._agent_ctxts = dict(tmp)

        self._server_worker = {
            "id": "s-0",
            "device": "",  # determined later in _set_server_device_id()
            "is_server": True,
            "stage": {"start": -1, "end": -1},
        }

        self._machine_to_agent_id: dict[int, str] = dict()

        # Keep track of machine and stage ID offsets
        self._machine_offsets: list[int] = []
        self._stage_id_offset = 0
        self._world_id_offset = 0

        self._server_starting_world_ids = []
        # Combined flow graph and workers
        self._combined_flow_graph = {"s-0": []}
        self._combined_workers = []
        self._combined_worker_to_gpu = {}
        self._combined_worker_to_machine = {}

        # machine to place the dispatcher
        self._server_machine = 0
        self._server_machine_ip = ""

        # All last stage replicas connections to the server
        self._final_server_connections = []

    def generate(self) -> JobConfig:
        """Generate a config."""
        self._set_offsets()

        self._map_machine_to_agent_id()

        config_data = self._process_multiple_exec_plans()

        config = JobConfig(**config_data)

        config = JobConfig.merge(self._base_cfg, config)

        config.validate()

        return config

    def _set_offsets(self) -> None:
        if self._base_cfg is None:
            return

        self._stage_id_offset = self._base_cfg.max_stage_id() + 1
        self._world_id_offset = self._base_cfg.max_world_id() + 1
        self._server_machine_ip = self._base_cfg.server_ip()

    def _process_multiple_exec_plans(self):
        """Process multiple execution configuration plans and generate a unified config."""
        micro_batch_size = 0
        for idx, plan in enumerate(self._plan_list):
            micro_batch_size = plan.batch_size

            # Process this pipeline config
            result = self._process_pipeline_config(idx, plan)

            # Update the combined flow graph
            for worker_id, connections in result["flow_graph"].items():
                if worker_id in self._combined_flow_graph:
                    self._combined_flow_graph[worker_id].extend(connections)
                else:
                    self._combined_flow_graph[worker_id] = connections

            # Add workers to combined list
            self._combined_workers.extend(result["workers"])
            self._combined_worker_to_gpu.update(result["worker_to_gpu"])
            self._combined_worker_to_machine.update(result["worker_to_machine"])

            # Update final server connections
            self._final_server_connections.extend(result["server_connections"])

            # Find max stage ID in this config
            max_stage_id = 0
            for worker in result["workers"]:
                if worker["id"] != "s-0":
                    stage_id = int(worker["id"].split("-")[0])
                    max_stage_id = max(max_stage_id, stage_id)

            self._stage_id_offset = max_stage_id + 1

            # Update world ID offset for next pipeline
            self._world_id_offset += result["total_world_ids"]
            # reserve one world id for connecting it to server
            self._server_starting_world_ids.append(self._world_id_offset)
            # increment world id offset by 1
            self._world_id_offset += 1

        return self._combine(micro_batch_size)

    def _set_server_device_id(self, agent_ctxt: AgentContext) -> None:
        if self._dispatcher_device != "cuda":
            self._server_worker["device"] = "cpu"
            return

        # Now add the server/dispatcher once for all pipelines
        # Check the number of GPU allocated on the final server machine
        server_machine_gpu_used = set()
        for worker in self._combined_worker_to_machine:
            if self._combined_worker_to_machine[worker] == self._server_machine:
                server_machine_gpu_used.add(self._combined_worker_to_gpu[worker])

        # Find the first available GPU on the final server machine
        for gpu_id in agent_ctxt.avail_gpus():
            if gpu_id not in server_machine_gpu_used:
                self._server_worker["device"] = f"cuda:{gpu_id}"
                return

        # if we reach here, server device is not set; so, raise an exception
        raise InsufficientResources("server's device not set")

    def _combine(self, micro_batch_size) -> dict:
        server_already_exists = True if self._server_machine_ip else False

        if not server_already_exists:
            # coming to this part means that this is the first time to generate
            # a config. In that case, we decide server machine's IP and device
            agent_id = self._machine_to_agent_id[self._server_machine]
            agent_ctxt = self._agent_ctxts[agent_id]

            self._server_machine_ip = agent_ctxt.ip
            self._set_server_device_id(agent_ctxt)

            # Add server at the beginning of the workers list
            self._combined_workers = [self._server_worker] + self._combined_workers

        for connection, world_id in zip(
            self._final_server_connections, self._server_starting_world_ids
        ):
            # We need to update the address and backend of the server connections
            connection["addr"] = self._server_machine_ip
            connection["name"] = JobConfig.world_name(world_id)

        self._combined_flow_graph["s-0"] = self._final_server_connections

        # Create the final config
        config = {
            "name": self._source.name,
            "model": self._source.model,
            "nfaults": self._source.nfaults,
            "micro_batch_size": micro_batch_size,
            "fwd_policy": self._source.fwd_policy,
            "job_id": self._source.job_id,
            "max_inflight": self._source.max_inflight,
            "flow_graph": self._combined_flow_graph,
            "dataset": self._source.dataset,
            "workers": self._combined_workers,
        }

        return config

    def _process_pipeline_config(self, idx: int, plan: ExecPlan) -> dict:
        """Process a single pipeline configuration."""
        stages = plan.stages

        # Generate unified allocation
        worker_to_machine, worker_to_gpu = self._map_worker_to_machine_gpu(idx, stages)

        # Create flow graph with world ID offset
        flow_graph, server_connections = self._create_flow_graph(
            stages, worker_to_machine
        )

        # Create workers
        workers = self._create_workers(stages, worker_to_machine, worker_to_gpu)

        # Count total world IDs used
        total_world_ids = 0
        for connections in flow_graph.values():
            total_world_ids += len(connections)

        return {
            "flow_graph": flow_graph,
            "workers": workers,
            "worker_to_machine": worker_to_machine,
            "worker_to_gpu": worker_to_gpu,
            "total_world_ids": total_world_ids,
            "server_connections": server_connections,
        }

    def _update_machine_worker_count(
        self, plan: ExecPlan, machine_offset: int, machine_worker_count: dict[int, int]
    ) -> int:
        executed_once = False
        max_machine_id = 0
        for stage in plan.stages:
            for machine_id_str, count in stage.gpu_allocation.items():
                executed_once = True

                machine_id = int(machine_id_str) + machine_offset
                machine_worker_count[machine_id] += count
                max_machine_id = max(max_machine_id, machine_id)

        if not executed_once:
            raise InvalidConfig("stages can't be empty")

        if max_machine_id >= len(self._agent_ctxts):
            err_msg = f"Machine ID {max_machine_id} is out of range for the number of machines ({len(self._agent_ctxts)})"
            raise InsufficientResources(err_msg)

        return max_machine_id

    def _map_machine_to_agent_id(self) -> None:
        machine_worker_count = defaultdict(int)

        machine_offset = 0
        for plan in self._plan_list:
            self._machine_offsets.append(machine_offset)
            max_machine_id = self._update_machine_worker_count(
                plan, machine_offset, machine_worker_count
            )
            machine_offset = max_machine_id + 1

        if len(machine_worker_count) > len(self._agent_ctxts):
            err_msg = f"need: {len(machine_worker_count)} nodes; available: {len(self._agent_ctxts)} nodes"
            raise InsufficientResources(err_msg)

        server_machine = None
        for mwc, ac in zip(machine_worker_count.items(), self._agent_ctxts.items()):
            mid, count = mwc[0], mwc[1]
            agent_id, agent_ctxt = ac[0], ac[1]

            avail_count = agent_ctxt.avail_gpu_count()
            if count > avail_count:
                err_msg = f"node {mid} needs {count} GPUs; agent {agent_id} has {avail_count} GPUs"
                raise InsufficientResources(err_msg)

            self._machine_to_agent_id[mid] = agent_id

            # determine server/dispatcher's machine
            if self._dispatcher_device == "cuda":
                if server_machine is None and avail_count - 1 >= count:
                    server_machine = mid
            else:
                if server_machine is None:
                    server_machine = mid

        if server_machine is None:
            err_msg = "server machine can't be set due to no GPU for it"
            raise InsufficientResources(err_msg)

        self._server_machine = server_machine

    def _map_worker_to_machine_gpu(
        self, idx: int, stages
    ) -> tuple[dict[str, int], dict[str, int]]:
        # Create worker ID to machine ID mapping
        worker_to_machine = {}
        worker_to_gpu = {}  # Maps worker ID to local GPU ID on the machine

        # Track already allocated GPUs per machine to assign local GPU IDs
        allocated_gpus = defaultdict(set)  # machine_id -> set of used local GPU ids

        # Directly assign each worker to a machine and GPU
        for stage in stages:
            orig_stage_id = stage.stage_id
            stage_id = orig_stage_id + self._stage_id_offset

            # Build a list of (machine_id, count) pairs for worker assignment
            total_count = 0
            machine_allocs = []
            for machine_id_str, count in stage.gpu_allocation.items():
                machine_id = int(machine_id_str) + self._machine_offsets[idx]
                machine_allocs.append((machine_id, count))
                total_count += count

            num_replicas = stage.num_replicas

            if total_count != num_replicas:
                err_msg = f"total # of required GPUs ({total_count}) is different from # of replicas ({num_replicas})"
                raise InvalidConfig(err_msg)

            # Assign workers to machines according to the allocation
            worker_idx = 0
            for machine_id, count in machine_allocs:
                agent_id = self._machine_to_agent_id[machine_id]
                agent_ctxt = self._agent_ctxts[agent_id]
                avail_gpus = agent_ctxt.avail_gpus()

                for _ in range(count):
                    wid = f"{stage_id}-{worker_idx}"
                    worker_to_machine[wid] = machine_id

                    found = False
                    # Find next available local GPU on this machine
                    for local_gpu in avail_gpus:
                        if local_gpu in allocated_gpus[machine_id]:
                            continue

                        worker_to_gpu[wid] = local_gpu
                        allocated_gpus[machine_id].add(local_gpu)
                        found = True
                        break

                    if not found:
                        # If we get here, we couldn't find an available GPU
                        err_msg = f"No GPU on node {machine_id} for worker {wid}"
                        raise InsufficientResources(err_msg)

                    worker_idx += 1

        return worker_to_machine, worker_to_gpu

    def _find_prev_stage(self, orig_stage_id: int, stages):
        # Find the previous stage
        prev = None
        prev_stage_id = None
        if orig_stage_id > 0:
            idx = orig_stage_id - 1
            stage = stages[idx]

            if stage.stage_id != idx:
                err_msg = f"stage id ({stage.stage_id}) must be the same as its index ({idx}) in the stages"
                raise InvalidConfig(err_msg)

            prev = stage
            prev_stage_id = prev.stage_id + self._stage_id_offset

        return prev, prev_stage_id

    def _create_flow_graph(self, stages, worker_to_machine):
        """Create flow graph configuration with distributed address mapping based on planning JSON stages."""
        flow_graph = {}
        current_world_id = self._world_id_offset
        server_backend = "gloo" if self._dispatcher_device == "cpu" else "nccl"

        # Add server connections
        server_connections = []

        # Get the last stage for connections
        last_stage = stages[-1]
        last_stage_id = last_stage.stage_id + self._stage_id_offset
        for r in range(last_stage.num_replicas):
            peer_id = f"{last_stage_id}-{r}"
            conn = {
                "name": None,
                "peers": FlowList([peer_id]),
                "addr": None,  # Server's own IP
                "backend": server_backend,
            }
            server_connections.append(conn)

        if "s-0" in flow_graph:
            err_msg = "server should not be in the flow graph"
            raise InvalidConfig(err_msg)

        # Add worker connections
        for stage in stages:
            orig_stage_id = stage.stage_id
            stage_id = orig_stage_id + self._stage_id_offset

            prev, prev_stage_id = self._find_prev_stage(orig_stage_id, stages)

            for r in range(stage.num_replicas):
                wid = f"{stage_id}-{r}"
                # Get worker's machine from the unified allocation
                worker_machine = worker_to_machine[wid]
                agent_id = self._machine_to_agent_id[worker_machine]
                agent_ctxt = self._agent_ctxts[agent_id]

                if orig_stage_id == 0:
                    peers = ["s-0"]
                    backend = server_backend
                else:
                    peers = [f"{prev_stage_id}-{i}" for i in range(prev.num_replicas)]
                    backend = "nccl"

                connections = []
                for peer in peers:
                    conn = {
                        "name": JobConfig.world_name(current_world_id),
                        "peers": FlowList([peer]),
                        "addr": agent_ctxt.ip,
                        "backend": backend if peer != "s-0" else server_backend,
                    }
                    connections.append(conn)
                    current_world_id += 1

                flow_graph[wid] = connections

        # Add feedback connections for llama generation
        if self._is_auto_regressive and len(stages) > 1:
            first_stage = stages[0]
            first_stage_id = first_stage.stage_id + self._stage_id_offset
            for r in range(first_stage.num_replicas):
                wid = f"{first_stage_id}-{r}"
                worker_machine = worker_to_machine[wid]
                agent_id = self._machine_to_agent_id[worker_machine]
                agent_ctxt = self._agent_ctxts[agent_id]
                for rr in range(last_stage.num_replicas):
                    peer = f"{last_stage_id}-{rr}"
                    conn = {
                        "name": f"w{current_world_id}",
                        "peers": FlowList([peer]),
                        "addr": agent_ctxt.ip,
                        "backend": "nccl",  # We have nccl connections between all workers
                    }
                    flow_graph[wid].append(conn)
                    current_world_id += 1

        return flow_graph, server_connections

    def _create_workers(self, stages, worker_to_machine, worker_to_gpu):
        """Create workers configuration with proper GPU assignments."""
        workers = []

        # Assign stage workers
        for stage in stages:
            orig_stage_id = stage.stage_id
            stage_id = orig_stage_id + self._stage_id_offset
            layer_start, layer_end = stage.layer_range

            for r in range(stage.num_replicas):
                wid = f"{stage_id}-{r}"
                local_gpu = worker_to_gpu[wid]

                worker = {
                    "id": wid,
                    "device": f"cuda:{local_gpu}",
                    "stage": {"start": layer_start, "end": layer_end},
                }
                workers.append(worker)

        return workers


class CfgGen2:
    """CfgGen class."""

    def __init__(
        self,
        placement: dict[str, any],
        agent_ctxts_list: list[AgentContext],
        source: JobConfig,
        dispatcher_device_type="cpu",
        base_cfg: JobConfig = None,
    ):
        """Initialize an instance."""
        self._placement = placement
        self._source = source
        self._dispatcher_device_type = dispatcher_device_type
        self._base_cfg = base_cfg

        self._is_auto_regressive = self._source.is_auto_regressive()

        self._agent_ctxts = {}
        for ctx in sorted(agent_ctxts_list, key=lambda ctx: ctx.id):
            self._agent_ctxts[ctx.id] = ctx

        self._dispatcher_worker = {
            "id": "s-0",
            "device": "",  # determined later
            "is_server": True,
            "stage": {"start": -1, "end": -1},
        }

        self._machine_to_agent_id: dict[int, str] = dict()

        # Track GPU allocation per node
        self._node_gpu_tracker = {}

        # variables for processing each deployment
        self._all_workers = []
        self._worker_to_machine = {}
        self._worker_to_gpu = {}
        self._deployment_workers = {}  # Track workers per deployment
        self._batch_size = 1

        self._stage_id_offset = 0
        self._world_id_offset = 0

        self._dispatcher_machine = ""
        self._dispatcher_addr = ""

    def generate(self) -> JobConfig:
        """Generate a config.

        Note: Do not modify the order of function calls. Some member
              variables updated in each function have dependencies
              in other functions.
        """
        self._precheck()

        self._set_offsets()

        self._map_machine_to_agent_id()

        self._assign_workers()

        self._set_dispatcher()

        # Build flow graph
        flow_graph = self._build_flow_graph()

        # BUild final config
        config_data = {
            "name": self._source.name,
            "model": self._source.model,
            "nfaults": self._source.nfaults,
            "micro_batch_size": self._batch_size,
            "fwd_policy": self._source.fwd_policy,
            "job_id": self._source.job_id,
            "max_inflight": self._source.max_inflight,
            "flow_graph": flow_graph,
            "dataset": self._source.dataset,
            "workers": self._all_workers,
        }

        config = JobConfig(**config_data)

        config = JobConfig.merge(self._base_cfg, config)

        config.validate()

        return config

    def _precheck(self) -> None:
        deployments = self._placement["deployments"]

        # Validate dispatcher exists
        if "dispatcher" not in deployments:
            raise RuntimeError("No dispatcher found in placement!")

    def _set_offsets(self) -> None:
        if self._base_cfg is None:
            return

        self._stage_id_offset = self._base_cfg.max_stage_id() + 1
        self._world_id_offset = self._base_cfg.max_world_id() + 1
        self._dispatcher_addr = self._base_cfg.server_ip()

    def _map_machine_to_agent_id(self) -> None:
        nodes = self._placement["nodes"]

        node_ids = []
        for node_id in nodes.keys():
            node_ids.append(int(node_id))
        node_ids = sorted(node_ids)

        if len(node_ids) != len(self._agent_ctxts):
            err_msg = f"{len(node_ids)} nodes != {len(self._agent_ctxts)} agents"
            raise DifferentResourceAmount(err_msg)

        for node_id, agent_id in zip(node_ids, self._agent_ctxts.keys()):
            self._machine_to_agent_id[node_id] = agent_id

    def _assign_workers(self) -> None:
        deployments = self._placement["deployments"]

        # Get all model deployments (excluding dispatcher)
        model_deployments = {
            deploy_id: info
            for deploy_id, info in deployments.items()
            if deploy_id != "dispatcher"
        }

        if not model_deployments:
            raise RuntimeError("No model deployments found!")

        # Track GPU allocation per node
        for key in self._placement["nodes"]:
            self._node_gpu_tracker[int(key)] = 0

        template_solutions = self._placement["template_solutions"]
        gpus_per_node = self._placement["meta"]["gpus_per_node"]

        # Process deployments in the exact order from JSON file
        for deploy_id, deploy_info in model_deployments.items():
            template_size = deploy_info["template_size"]
            template_path = template_solutions[str(template_size)]

            # Load template info
            template_info = self._load_template_info(template_path)
            self._batch_size = template_info["batch_size"]

            # Allocate workers for this deployment
            workers, w_to_m, w_to_g = self._allocate_workers_for_deployment(
                deploy_id, deploy_info, template_info, gpus_per_node
            )

            self._deployment_workers[deploy_id] = workers
            self._all_workers.extend(workers)
            self._worker_to_machine.update(w_to_m)
            self._worker_to_gpu.update(w_to_g)

            # Update stage_id_offset for next deployment
            max_stage_id = max(int(w["id"].split("-")[0]) for w in workers)
            self._stage_id_offset = max_stage_id + 1

    def _set_dispatcher(self) -> None:
        if self._dispatcher_addr:
            print("Do nothing; dispatcher already configured")
            return

        deployments = self._placement["deployments"]

        # Extract dispatcher info
        dispatcher_info = deployments["dispatcher"]
        self._dispatcher_machine = dispatcher_info["node_segments"][0]["node_id"]

        dispatcher_gpu = (
            self._node_gpu_tracker[self._dispatcher_machine]
            if self._dispatcher_device_type == "cuda"
            else None
        )

        dispatcher_device = (
            f"cuda:{dispatcher_gpu}"
            if self._dispatcher_device_type == "cuda"
            else "cpu"
        )

        self._dispatcher_worker["device"] = dispatcher_device

        # Add dispatcher worker
        self._all_workers.insert(0, self._dispatcher_worker)  # Server first

    def _load_template_info(self, template_path: str) -> dict:
        """Load template JSON and extract stage information."""
        with open(template_path) as f:
            data = json.load(f)

        stages = data.get("stages", data.get("pipeline_stages", []))
        batch_size = data.get("batch_size", 1)

        return {"stages": stages, "batch_size": batch_size}

    def _allocate_workers_for_deployment(
        self,
        deploy_id,
        deploy_info,
        template_info,
        gpus_per_node,
    ):
        """
        Allocate workers for a single deployment based on its node_segments.

        Returns:
            workers: list of worker dicts
            worker_to_machine: dict mapping worker_id to node_id
            worker_to_gpu: dict mapping worker_id to local_gpu_id
        """
        node_segments = deploy_info["node_segments"]
        stages = template_info["stages"]

        # TODO(MLEE): need to identify available GPU IDs via agent contexts
        #             to make gpu id assignment robust
        # Create a flat list of GPUs available from node_segments
        available_gpus = []  # List of (node_id, local_gpu_id) tuples
        for segment in node_segments:
            node_id = segment["node_id"]
            gpus_in_segment = segment["gpus"]

            # Get the starting local GPU ID for this node
            start_local_gpu = self._node_gpu_tracker[node_id]

            if gpus_in_segment + start_local_gpu > gpus_per_node:
                err_msg = f"GPUs in segment {gpus_in_segment} > # of available GPUs {gpus_per_node}"
                raise InsufficientResources(err_msg)

            # Add GPUs from this segment
            for i in range(gpus_in_segment):
                local_gpu = start_local_gpu + i
                available_gpus.append((node_id, local_gpu))

            # Update the tracker
            self._node_gpu_tracker[node_id] += gpus_in_segment

        # Now allocate workers based on stages and their gpu_allocation
        workers = []
        worker_to_machine = {}
        worker_to_gpu = {}
        gpu_idx = 0  # Index into available_gpus

        for stage in stages:
            stage_id = stage["stage_id"] + self._stage_id_offset
            layer_start, layer_end = stage["layer_range"]
            num_replicas = stage["num_replicas"]
            gpu_allocation = stage["gpu_allocation"]

            # Allocate replicas for this stage
            replica_idx = 0
            for machine_offset_str, gpu_count in gpu_allocation.items():
                for _ in range(gpu_count):
                    if replica_idx < num_replicas and gpu_idx < len(available_gpus):
                        worker_id = f"{stage_id}-{replica_idx}"
                        node_id, local_gpu = available_gpus[gpu_idx]

                        workers.append(
                            {
                                "id": worker_id,
                                "device": f"cuda:{local_gpu}",
                                "stage": {"start": layer_start, "end": layer_end},
                            }
                        )

                        worker_to_machine[worker_id] = node_id
                        worker_to_gpu[worker_id] = local_gpu

                        replica_idx += 1
                        gpu_idx += 1

        return workers, worker_to_machine, worker_to_gpu

    def _build_flow_graph(self):
        """Build the flow graph for communication."""
        flow_graph = {}
        world_id = self._world_id_offset
        server_backend = "nccl" if self._dispatcher_device_type == "cuda" else "gloo"

        # Process each deployment separately
        for deploy_id, deploy_workers in self._deployment_workers.items():
            # Group workers by stage_id within this deployment
            stages = {}
            for worker in deploy_workers:
                stage_id = int(worker["id"].split("-")[0])
                if stage_id not in stages:
                    stages[stage_id] = []
                stages[stage_id].append(worker)

            stage_ids = sorted(stages.keys())

            # Build connections for this deployment's pipeline
            for i, stage_id in enumerate(stage_ids):
                stage_workers = stages[stage_id]

                for worker in stage_workers:
                    worker_id = worker["id"]
                    worker_node = self._worker_to_machine[worker_id]
                    agent_id = self._machine_to_agent_id[worker_node]
                    ctx = self._agent_ctxts[agent_id]
                    worker_addr = ctx.ip

                    connections = []

                    # Connection to previous stage or server
                    if i == 0:  # First stage connects to server
                        connections.append(
                            {
                                "name": f"w{world_id}",
                                "peers": FlowList(["s-0"]),
                                "addr": worker_addr,
                                "backend": server_backend,
                            }
                        )
                        world_id += 1
                    else:  # Connect to all workers in previous stage
                        prev_stage_id = stage_ids[i - 1]
                        prev_workers = stages[prev_stage_id]
                        for prev_worker in prev_workers:
                            connections.append(
                                {
                                    "name": f"w{world_id}",
                                    "peers": FlowList([prev_worker["id"]]),
                                    "addr": worker_addr,
                                    "backend": "nccl",
                                }
                            )
                            world_id += 1

                    # For LLM(e.g., llama), add feedback connections from last stage to first stage
                    if self._is_auto_regressive and i == 0 and len(stage_ids) > 1:
                        last_stage_workers = stages[stage_ids[-1]]
                        for last_worker in last_stage_workers:
                            connections.append(
                                {
                                    "name": f"w{world_id}",
                                    "peers": FlowList([last_worker["id"]]),
                                    "addr": worker_addr,
                                    "backend": "nccl",
                                }
                            )
                            world_id += 1

                    flow_graph[worker_id] = connections

        # Add server connections - connect to last stage of each deployment
        server_connections = []
        for deploy_workers in self._deployment_workers.values():
            # Find last stage workers in this deployment
            max_stage_id = max(int(w["id"].split("-")[0]) for w in deploy_workers)
            last_stage_workers = [
                w for w in deploy_workers if int(w["id"].split("-")[0]) == max_stage_id
            ]

            # dispatcher address is not set yet
            if not self._dispatcher_addr:
                agent_id = self._machine_to_agent_id[self._dispatcher_machine]
                ctx = self._agent_ctxts[agent_id]
                self._dispatcher_addr = ctx.ip

            for worker in last_stage_workers:
                server_connections.append(
                    {
                        "name": f"w{world_id}",
                        "peers": FlowList([worker["id"]]),
                        "addr": self._dispatcher_addr,
                        "backend": server_backend,
                    }
                )
                world_id += 1

        flow_graph["s-0"] = server_connections

        return flow_graph
