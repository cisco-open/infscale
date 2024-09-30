from dataclasses import dataclass
from enum import Enum

from infscale.config import ServeConfig


class MessageType(Enum):
    """MessageType enum."""

    LOG = "log"
    TERMINATE = "terminate"
    STATUS = "status"
    CONFIG = "config"


class WorkerStatus(Enum):
    """WorkerStatus enum"""

    READY = "ready"
    STARTED = "started"
    RUNNING = "running"
    DONE = "done"
    TERMINATED = "terminated"
    FAILED = "failed"


@dataclass
class Message:
    """Message dataclass."""

    type: MessageType
    content: str | WorkerStatus | ServeConfig
