"""Class to keep world information."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WorldInfo:
    """Information about World.

    Currently we only consider a world with one leader and one worker.
    Since there are two processes in each world, rank is either 0 or 1.
    """

    name: str  # world's name
    me: int  # my rank
    other: int  # other peer's rank
