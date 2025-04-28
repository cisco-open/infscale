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

    RUN = "run"
    STATUS = "status"
    CLEANUP = "cleanup"


@dataclass
class CommandConfig:
    env_activate_command: str
    work_dir: str
    log_level: str
    action: str
    entity: str
    extra_args: str = ""

    def __str__(self) -> str:
        """Render shell command from a mustache template."""
        template = Path("templates/shell_command.sh.mustache").read_text()
        rendered = pystache.render(template, self)

        return rendered


@dataclass
class TaskConfig:
    """Class for defining test task config."""

    name: str
    shell: str
    work_dir: str
    env_activate_command: str
    log_level: str

    def __post_init__(self):
        self.shell = str(
            CommandConfig(
                **self.shell,
                work_dir=self.work_dir,
                env_activate_command=self.env_activate_command,
                log_level=self.log_level,
            )
        )

    def __str__(self) -> None:
        """Render task from a mustache template."""
        template = Path("templates/task.yml").read_text()
        rendered = pystache.render(template, self)

        return rendered


@dataclass
class TestStep:
    """Class for defining test step."""

    name: str
    work_dir: str
    env_activate_command: str
    log_level: str
    host: str = "all"
    tasks: list[str] = ""
    job_id: str = ""
    statuses: str = ""
    type: TestType = TestType.RUN

    def __post_init__(self):
        if not self.tasks:
            return
        
        for i, task in enumerate(self.tasks):
            task_config = TaskConfig(
                        **task,
                        work_dir=self.work_dir,
                        env_activate_command=self.env_activate_command,
                        log_level=self.log_level,
                    )
            self.tasks[i] = str(task_config)

    def __str__(self) -> None:
        """Render config from a mustache template."""
        rendered = None
        type_enum = TestType(self.type)

        match type_enum:
            case TestType.RUN:
                template = Path("templates/play.yml").read_text()
                rendered_tasks = "\n".join(indent(task, 4) for task in self.tasks)
                render_data = {
                        "name": self.name,
                        "host": self.host,
                        "tasks": rendered_tasks,
                    }
                rendered = pystache.render(
                    template,
                    render_data,
                )
            case TestType.STATUS:
                template = Path("templates/job_status.yml").read_text()
                rendered_tasks = "\n".join(indent(task, 4) for task in self.tasks)
                render_data = {
                        "host": self.host,
                        "statuses": self.statuses,
                        "job_id": self.job_id,
                    }
                rendered = pystache.render(
                    template,
                    render_data,
                )
            case TestType.CLEANUP:
                template = Path("templates/cleanup_processes.yml").read_text()
                rendered_tasks = "\n".join(indent(task, 4) for task in self.tasks)
                rendered = pystache.render(template, {"name": self.name})

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
            self.steps[i] = test_step


def indent(text: str, spaces: int = 4) -> str:
    """Indent string in accepted YAML format."""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )
