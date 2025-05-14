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

import json
from pathlib import Path

from infscale.configs.job import JobConfig
from infscale.configs.plan import ExecPlan
from infscale.controller.agent_context import AgentContext
from infscale.controller.cfggen import CfgGen


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

        self._plans = sorted(self._plans, key=lambda plan: plan.pipeline_throughput)

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


class Planner:
    """Planner class."""

    def __init__(self, path: str, autoscale: bool) -> None:
        """Initialize instance."""
        self._path = Path(path).expanduser()

        self._autoscale = autoscale

        self._colls: dict[str, PlanCollection] = {}

    def build_config(
        self, source: JobConfig, agent_ctxts: dict[str, AgentContext], demand: float = 0
    ) -> JobConfig:
        """Build a config based on source config."""
        if not self._autoscale:
            # if autoscale is not enabled, we use source as is
            return source

        self._load_plans(source.model)

        # configure plan collection to set a subset of execution plans to be considered
        plan_list = self._colls[source.model].pick_plans(demand)
        gen = CfgGen(agent_ctxts, source, plan_list, "cuda")
        return gen.generate()

    def _load_plans(self, model_name: str) -> None:
        if model_name in self._colls:
            return

        self._colls[model_name] = PlanCollection()

        model_plan_path = self._path / model_name
        for entry in model_plan_path.iterdir():
            if not entry.is_file():
                continue

            self._colls[model_name].add(entry.absolute())

        # sort a plan collection for the model in an increasing order of throughput
        self._colls[model_name].sort()
