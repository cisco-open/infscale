from dataclasses import dataclass
from enum import Enum


class MessageType(Enum):
    """MessageType enum."""

    LOG = "log"
    TERMINATE = "terminate"
    STATUS = "status"


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
    content: str | WorkerStatus
