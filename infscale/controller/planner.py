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

"""planner.py."""

from dataclasses import dataclass
import json
from pathlib import Path

from infscale.common.exceptions import InsufficientResources
from infscale.configs.job import JobConfig
from infscale.configs.plan import ExecPlan
from infscale.controller.agent_context import AgentContext
from infscale.controller.cfggen import CfgGen, CfgGen2
from infscale.controller.deployment_placement import Placement


class PlanCollection:
    """PlanCollection class."""

    def __init__(self):
        """Initialize an instance."""
        self._plans: list[ExecPlan] = []

    def add(self, json_file: str) -> None:
        """Add pipeline stats to the collection."""
        # Read JSON file
        path = Path(json_file).expanduser()
        with open(path.absolute(), "r") as f:
            json_data = json.load(f)

        plan = ExecPlan.from_json(json_data)
        self._plans.append(plan)

    def sort(self) -> None:
        """Sort the exec plan list by throughput."""
        self._plans[0]

        self._plans = sorted(self._plans, key=lambda plan: plan.throughput)

    def pick_plans(self, demand: float = 0) -> list[ExecPlan]:
        """
        Return necessary plans to meed the demand.

        Attributes:
            demand (float): the number of requests / seconds.
        """
        idx = 0
        candidates = []
        capacity = -(10**-10)  # a small negative number  at least to pick one plan

        # demand should be at least zero
        demand = max(demand, 0.0)

        # TODO: need sophisticated algorithm
        while capacity < demand:
            plan = self._plans[idx]
            candidates.append(plan)
            capacity += plan.throughput

            idx = (idx + 1) % len(self._plans)

        return candidates

    def enumerate(self) -> ExecPlan:
        """Enumerate each exec plan."""
        for plan in self._plans:
            yield plan


@dataclass
class PipelineData:
    """PipelineData class."""

    worker_ids: set[str]
    total_throughput: float


class Planner:
    """Planner class."""

    def __init__(self, path: str, autoscale: bool) -> None:
        """Initialize instance."""
        self._path = Path(path).expanduser()

        self._autoscale = autoscale

        self._colls: dict[str, PlanCollection] = {}
        
        self.pipeline_data: dict[str, list[PipelineData]] = {}

    def build_config(
        self,
        source: JobConfig,
        agent_ctxts: dict[str, AgentContext],
        demand: float = 0,
        base_cfg: JobConfig = None,
    ) -> JobConfig:
        """Build a config based on source config."""
        if not self._autoscale:
            # if autoscale is not enabled, we use source as is
            return source

        # if base_cfg is none, this is the first time we build a config,
        # so we need to place the dispatcher on a GPU
        # otherwise, we already have a base config, so we don't need to
        # spare a GPU for the dispatcher
        dispatcher_on_gpu = base_cfg is None
        solution = self._calculate_placement(
            source, agent_ctxts, demand, dispatcher_on_gpu=dispatcher_on_gpu
        )

        if solution is None:
            raise InsufficientResources("No placement solution found")

        placement, agent_ctxts_list, total_throughput = solution

        gen2 = CfgGen2(placement, agent_ctxts_list, source, "cuda", base_cfg)
        cfg = gen2.generate()

        self._set_pipeline_data(cfg, total_throughput)

        return cfg

        #####
        # NOTE: disabled CfgGen for now; once CfgGen2 becomes stable, remove CfgGen
        #####

        # self._load_plans(source.model)

        # # configure plan collection to set a subset of execution plans to be considered
        # plan_list = self._colls[source.model].pick_plans(demand)
        # gen = CfgGen(agent_ctxts, source, plan_list, "cuda", base_cfg)
        # return gen.generate()
        
    def _set_pipeline_data(self, cfg: JobConfig, total_throughput) -> None:
        """Set pipeline data."""
        job_id = cfg.job_id

        if job_id not in self.pipeline_data:
            self.pipeline_data[job_id] = []

        pipeline_identifiers = JobConfig.get_pipeline_identifiers(cfg)
        prev_identifiers = {wid for data in self.pipeline_data[job_id] for wid in data.worker_ids}
        new_identifiers = pipeline_identifiers - prev_identifiers

        self.pipeline_data[job_id].append(PipelineData(new_identifiers, total_throughput))

    def _search_feasible_placement(
        self,
        demand: float,
        nfaults: int,
        placement: Placement,
        gpu_count: int,
        ctx_list: list[AgentContext],
        dispatcher_on_gpu: bool = True,
    ) -> tuple[dict, list[AgentContext], float] | None:
        # we'd like to search a feasible solution by increasing the number of nodes
        for num_nodes in range(1, len(ctx_list) + 1):
            res = placement.calculate_placement(
                gpu_count, len(ctx_list[:num_nodes]), nfaults, dispatcher_on_gpu
            )
            meta = res["meta"]
            if meta["total_throughput"] > demand:
                return (res, ctx_list[:num_nodes], meta["total_throughput"])

        return None

    def _calculate_placement(
        self,
        source: JobConfig,
        agent_ctxts: dict[str, AgentContext],
        demand: float,
        dispatcher_on_gpu: bool = True,
    ) -> tuple[dict, list[AgentContext], float] | None:
        gpu_count_and_nodes: dict[int, list[AgentContext]] = {}
        for ctx in agent_ctxts.values():
            count = ctx.avail_gpu_count()
            if count == 0:
                continue

            if count not in gpu_count_and_nodes:
                gpu_count_and_nodes[count] = []
            gpu_count_and_nodes[count].append(ctx)

        solution = None
        p = Placement(self._path / source.model.lower())
        for gpu_count in sorted(gpu_count_and_nodes.keys()):
            ctx_list = gpu_count_and_nodes[gpu_count]
            solution = self._search_feasible_placement(
                demand, source.nfaults, p, gpu_count, ctx_list, dispatcher_on_gpu
            )
            if solution:
                break

        return solution

    def _load_plans(self, model_name: str) -> None:
        if model_name in self._colls:
            return

        self._colls[model_name] = PlanCollection()

        model_plan_path = self._path / model_name.lower()
        for entry in model_plan_path.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".json":
                continue

            self._colls[model_name].add(entry.absolute())

        # sort a plan collection for the model in an increasing order of throughput
        self._colls[model_name].sort()
