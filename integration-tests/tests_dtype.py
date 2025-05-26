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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pystache
from constants import LOG_FOLDER


class CmdType(Enum):
    """CmdType enum."""

    INFSCALE_CMD = "infscale_cmd"
    OTHER = "other"


@dataclass
class TestController:
    host: str
    ip: str

    def __post_init__(self):
        if not self.host:
            raise ValueError(f"Controller host is required.")


@dataclass
class TestAgent:
    host: str


@dataclass
class TestControllerConfig:
    policy: str = ""
    config: str = ""


@dataclass
class TestConfig:
    controller: TestController
    agents: list[TestAgent]

    def __post_init__(self) -> None:
        if len(self.agents) == 0:
            raise ValueError(f"Running tests require at least one agent.")

        if not self.controller:
            raise ValueError(f"Controller config is required.")

        self.controller = TestController(**self.controller)

        for i, agent in enumerate(list(self.agents)):
            self.agents[i] = TestAgent(**agent)

    def get_hosts(self) -> list[str]:
        """Return a list of hosts used in this test config."""
        hosts = []

        hosts.append(self.controller.host)

        for agent in self.agents:
            hosts.append(agent.host)

        return hosts


@dataclass
class CommandConfig:
    env_activate_command: str
    work_dir: str
    log_level: str
    cmd: str
    args: str
    type: str

    def __post_init__(self):
        self.infscale_cmd = self.type == CmdType.INFSCALE_CMD
        self.background_run = "job" not in self.cmd
        self.log_folder = LOG_FOLDER

    def __str__(self) -> str:
        """Render shell command from a mustache template."""
        tpl_path = "templates/infscale_shell_command.mustache"

        if self.infscale_cmd and self.background_run:
            tpl_path = "templates/infscale_async_shell_command.mustache"

        if not self.infscale_cmd:
            tpl_path = "templates/other_shell_command.mustache"

        template = Path(tpl_path).read_text()

        rendered = pystache.render(template, self)

        return rendered


@dataclass
class ProcessCondition:
    success: str
    fail: str


@dataclass
class ProcessConfig:
    """Class for defining test process config."""

    cmd: str
    work_dir: str
    env_activate_command: str
    log_level: str
    type: CmdType = CmdType.INFSCALE_CMD
    args: str = ""
    condition: ProcessCondition = None

    def __post_init__(self):
        if self.condition:
            self.condition = ProcessCondition(**self.condition)

        self.wait_response = bool(self.condition)
        self.shell = str(
            CommandConfig(
                cmd=self.cmd,
                args=self.args,
                type=self.type,
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
            )
        )

    def __str__(self) -> None:
        """Render task from a mustache template."""
        template = Path("templates/task.yaml").read_text()
        rendered = pystache.render(template, self)
        return rendered


@dataclass
class TestStep:
    """Class for defining test step."""

    work_dir: str
    env_activate_command: str
    log_level: str
    processes: str = ""
    rendered_processes = []
    host: str = "all"

    def __post_init__(self):
        if not self.processes:
            return
        self.rendered_processes = list(self.processes)

        for i, process in enumerate(self.processes):
            process_cfg = ProcessConfig(
                **process,
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
            )
            self.rendered_processes[i] = str(process_cfg)

    def __str__(self) -> None:
        """Render config from a mustache template."""
        template = Path("templates/play.yaml").read_text()
        rendered_tasks = "\n".join(
            indent(process, 4) for process in self.rendered_processes
        )

        render_data = {
            "name": "Running test",
            "host": self.host,
            "tasks": rendered_tasks,
        }
        rendered = pystache.render(
            template,
            render_data,
        )
        return rendered


@dataclass
class Test:
    """Class for defining test case."""

    config: TestConfig
    work_dir: str
    env_activate_command: str
    log_level: str
    steps: list[TestStep]
    timeout: int = 60  # test timeout in seconds
    controller: TestControllerConfig = None

    def __post_init__(self):
        if self.controller:
            self.controller = TestControllerConfig(**self.controller)

        ctrl_host = self.config.controller.host
        steps = []

        self._add_ctrl_step(steps)
        self._add_agent_steps(steps)

        for i, step in enumerate(list(self.steps)):
            test_step = TestStep(
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
                host=ctrl_host,
                **step,
            )
            steps.append(test_step)

        self.steps = steps

    def _add_agent_steps(self, steps: list[TestStep]) -> None:
        """Add agents test steps."""
        for agent_config in self.config.agents:
            step_dict = {
                "host": agent_config.host,
                "processes": [
                    {
                        "cmd": "start agent",
                        "args": f"{agent_config.host} --host {self.config.controller.ip}",
                    }
                ],
            }

            agent_step = TestStep(
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
                **step_dict,
            )

            steps.append(agent_step)

    def _build_ctrl_args(self) -> str:
        """Build controller args string based on controller cfg."""
        args = ""
        policy, config = self.controller.policy, self.controller.config
        if policy:
            args += f"--policy {policy} "

        if config:
            args += f"--config {config} "

        return args

    def _add_ctrl_step(self, steps: list[TestStep]) -> None:
        """Prepend controller test step."""
        args = ""
        if self.controller:
            args = self._build_ctrl_args()

        ctrl_step = {
            "host": self.config.controller.host,
            "processes": [
                {
                    "cmd": "start controller",
                    "args": args,
                }
            ],
        }

        ctrl_step = TestStep(
            work_dir=self.work_dir,
            env_activate_command=self.env_activate_command,
            log_level=self.log_level,
            **ctrl_step,
        )

        steps.append(ctrl_step)


def indent(text: str, spaces: int = 4) -> str:
    """Indent string in accepted YAML format."""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )
