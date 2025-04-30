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

from collections import defaultdict

import yaml

from infscale.common.exceptions import InsufficientResources
from infscale.configs.job import JobConfig
from infscale.configs.plan import ExecPlan
from infscale.controller.agent_context import AgentContext


class FlowList(list):  # noqa: D101
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
    ):
        """Initialize an instance."""
        self._source = source
        self._plan_list = plan_list
        self._dispatcher_device = dispatcher_device

        # key: agent id and value is AgentContext
        # sort agent context by # of unused gpus in a decreasing order
        # and then by agent id in an increasing order to break a tie
        tmp = sorted(
            agent_ctxts.items(), key=lambda item: (-item[1].avail_gpu_count(), item[0])
        )
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

        # Combined flow graph and workers
        self._combined_flow_graph = {"s-0": []}
        self._combined_workers = []
        self._combined_worker_to_gpu = {}
        self._combined_worker_to_machine = {}

        # machine to place the dispatcher
        self._server_machine = 0

        # All last stage replicas connections to the server
        self._final_server_connections = []

    def generate(self) -> JobConfig:
        """Generate a config."""
        self._map_machine_to_agent_id()

        config_data = self._process_multiple_exec_plans()
        config = JobConfig(**config_data)
        config.validate()

        return config

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

        assert False, "server's device must be set"

    def _combine(self, micro_batch_size) -> dict:
        agent_id = self._machine_to_agent_id[self._server_machine]
        agent_ctxt = self._agent_ctxts[agent_id]

        self._set_server_device_id(agent_ctxt)

        # Add server at the beginning of the workers list
        self._combined_workers = [self._server_worker] + self._combined_workers

        # Add server to the flow graph
        server_starting_world_id = self._world_id_offset

        for connection in self._final_server_connections:
            # We need to update the address and backend of the server connections
            connection["addr"] = agent_ctxt.ip
            connection["name"] = f"w{server_starting_world_id}"
            server_starting_world_id += 1
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

        assert executed_once, "stages can't be empty"

        assert_msg = f"Machine ID {max_machine_id} is out of range for the number of machines ({len(self._agent_ctxts)})"
        assert max_machine_id < len(self._agent_ctxts), assert_msg

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

        # key: node id and value is # of required gpus
        # sort the dictionary by # of required gpus in a decreasing order
        # and then by node id in an increasing order to break a tie
        tmp = sorted(machine_worker_count.items(), key=lambda item: (-item[1], item[0]))
        machine_worker_count = dict(tmp)

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
                if avail_count - 1 >= count:
                    server_machine = mid
            else:
                server_machine = mid

        assert_msg = "server machine can't be set due to no GPU for it"
        assert server_machine is not None, assert_msg

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
            assert_msg = f"total # of required GPUs ({total_count}) is different from # of replicas ({num_replicas})"
            assert total_count == num_replicas, assert_msg

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

            assert_msg = f"stage id ({stage.stage_id}) must be the same as its index ({idx}) in the stages"
            assert stage.stage_id == idx, assert_msg

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

        assert "s-0" not in flow_graph, "Server should not be in the flow graph"

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
                        "name": f"w{current_world_id}",
                        "peers": FlowList([peer]),
                        "addr": agent_ctxt.ip,
                        "backend": backend if peer != "s-0" else server_backend,
                    }
                    connections.append(conn)
                    current_world_id += 1

                flow_graph[wid] = connections

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
