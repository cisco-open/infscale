"""Timer class."""

import asyncio
from asyncio import Task
from typing import Any, Callable

from infscale import get_logger

logger = get_logger()


class Timer:
    """Timer Class."""

    def __init__(self, timeout: int, callback: Callable, *args: Any, **kwargs: Any):
        """Initialize timer instance.

        *args and **kwargs are parameters for callback
        """
        self._timeout = timeout
        self._callback = callback
        self._args = args
        self._kwargs = kwargs

        self._task = self._create()

    def _create(self) -> Task:
        return asyncio.create_task(self._wait())

    async def _wait(self) -> None:
        logger.debug(f"wait for {self._timeout} seconds")
        await asyncio.sleep(self._timeout)

        if asyncio.iscoroutinefunction(self._callback):
            logger.debug("call coroutine callback")
            await self._callback(*self._args, **self._kwargs)
        else:
            logger.debug("call normal callback")
            self._callback(*self._args, **self._kwargs)

    def cancel(self) -> None:
        """Cancel the timer."""
        if self._task is None or self._task.cancelled():
            return

        self._task.cancel()

    def renew(self) -> None:
        """Renew the timer."""
        self.cancel()

        self._task = self._create()
