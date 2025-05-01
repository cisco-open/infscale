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


class TestType(Enum):
    """TestType enum."""

    START = "start"
    UPDATE = "update"
    JOB_STATUS = "job_status"


@dataclass
class CommandConfig:
    env_activate_command: str
    work_dir: str
    log_level: str
    type: str
    process: str = ""

    def __str__(self) -> str:
        """Render shell command from a mustache template."""
        template = Path("templates/shell_command.mustache").read_text()
        rendered = pystache.render(template, self)

        return rendered


@dataclass
class ProcessConfig:
    """Class for defining test process config."""

    process: str
    type: str
    work_dir: str
    env_activate_command: str
    log_level: str

    def __post_init__(self):
        self.shell = str(
            CommandConfig(
                process=self.process,
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
    type: TestType = TestType.START
    processes: str = ""
    rendered_processes = []
    statuses: str = ""
    host: str = "all"
    job_id: str = ""


    def __post_init__(self):
        if not self.processes:
            return
        self.rendered_processes = list(self.processes)
        for i, process in enumerate(self.processes):
            process_cfg = ProcessConfig(
                        type=self.type,
                        process=f"{self.type} {process}",
                        work_dir=self.work_dir,
                        env_activate_command=self.env_activate_command,
                        log_level=self.log_level,
                    )
            self.rendered_processes[i] = str(process_cfg)

    def __str__(self) -> None:
        """Render config from a mustache template."""
        rendered = None
        type_enum = TestType(self.type)

        match type_enum:
            case TestType.START | TestType.UPDATE:
                template = Path("templates/play.yaml").read_text()
                rendered_tasks = "\n".join(indent(process, 4) for process in self.rendered_processes)

                render_data = {
                        "name": f"{self.type} {", ".join(self.processes)}",
                        "host": self.host,
                        "tasks": rendered_tasks,
                    }
                rendered = pystache.render(
                    template,
                    render_data,
                )

            case TestType.JOB_STATUS:
                template = Path("templates/job_status.yaml").read_text()
                render_data = {
                        "host": self.host,
                        "statuses": self.statuses,
                        "job_id": self.job_id,
                    }
                rendered = pystache.render(
                    template,
                    render_data,
                )

        return rendered


@dataclass
class TestConfig:
    """Class for defining test config."""

    work_dir: str
    env_activate_command: str
    log_level: str
    steps: list[str]

    def __post_init__(self):
        for i, step in enumerate(self.steps):
            test_step = TestStep(
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
                **step,
            )
            self.steps[i] = str(test_step)


def indent(text: str, spaces: int = 4) -> str:
    """Indent string in accepted YAML format."""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )
