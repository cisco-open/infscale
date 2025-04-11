from dataclasses import dataclass
from pathlib import Path

import pystache


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

    def __post_init__(self):
        self.shell = str(CommandConfig(**self.shell))

    def __str__(self) -> None:
        """Render task from a mustache template."""
        template = Path("templates/task.yml").read_text()
        rendered = pystache.render(template, self)

        return rendered


@dataclass
class TestConfig:
    """Class for defining test config."""

    name: str
    host: str
    tasks: list[str]

    def __post_init__(self):
        self.tasks = [str(TaskConfig(**task)) for task in self.tasks]

    def __str__(self) -> None:
        """Render config from a mustache template."""
        template = Path("templates/play.yml").read_text()
        rendered_tasks = "\n".join(indent(task, 4) for task in self.tasks)
        rendered = pystache.render(
            template, {"name": self.name, "host": self.host, "tasks": rendered_tasks}
        )

        return rendered


def indent(text: str, spaces: int = 4) -> str:
    """Indent string in accepted YAML format."""
    prefix = " " * spaces
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )
