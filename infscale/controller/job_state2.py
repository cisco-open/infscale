from __future__ import annotations
from enum import Enum


class InvalidJobStateAction(Exception):
    """
    Custom exception for invalid actions in a job state.
    """

    def __init__(self, job_id, action, state):
        self.job_id = job_id
        self.action = action
        self.state = state
        super().__init__(
            f"Job {job_id}: '{action}' action is not allowed in the '{state}' state."
        )


class JobStateEnum(Enum):
    """JobState enum."""

    READY = "ready"
    RUNNING = "running"
    STARTING = "starting"
    STOPPED = "stopped"
    STOPPING = "stopping"
    UPDATING = "updating"


class BaseJobState():
    """Abstract base class for job states."""

    def __init__(self, context: JobStateContext):
        self.context = context
        self.job_id = context.job_id

    def start(self):
        """Transition to STARTING state."""
        raise InvalidJobStateAction(self.job_id, "start", self.context.state_enum.value)

    def stop(self):
        """Transition to STOPPING state."""
        raise InvalidJobStateAction(self.job_id, "stop", self.context.state_enum.value)

    def update(self):
        """Transition to UPDATING state."""
        raise InvalidJobStateAction(
            self.job_id, "update", self.context.state_enum.value
        )

    def cond_running(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "running", self.context.state_enum.value
        )

    def cond_updated(self):
        """Handle the transition to running."""
        raise InvalidJobStateAction(
            self.job_id, "updating", self.context.state_enum.value
        )

    def cond_stopped(self):
        """Handle the transition to stopped."""
        raise InvalidJobStateAction(
            self.job_id, "stopping", self.context.state_enum.value
        )


class ReadyState(BaseJobState):
    def start(self):
        """Transition to STARTING state."""
        print("Starting job...")
        self.job.set_state(JobStateEnum.STARTING)


class RunningState(BaseJobState):
    """RunningState class."""

    def stop(self):
        """Transition to STOPPING state."""
        self.job.set_state(JobStateEnum.STOPPING)

    def update(self):
        """Transition to UPDATING state."""
        self.job.set_state(JobStateEnum.UPDATING)


class StartingState(BaseJobState):
    """StartingState class."""

    def stop(self):
        """Transition to STOPPING state."""
        print("Stopping job...")
        self.job.set_state(JobStateEnum.STOPPING)

    def cond_running(self):
        """Handle the transition to running."""
        self.job.set_state(JobStateEnum.RUNNING)


class StoppedState(BaseJobState):
    """StoppedState class."""

    def start(self):
        """Transition to STARTING state."""
        self.job.set_state(JobStateEnum.STARTING)


class StoppingState(BaseJobState):
    """StoppingState class."""

    def cond_stopped(self):
        """Handle the transition to stopped."""
        self.job.set_state(JobStateEnum.STOPPED)


class UpdatingState(BaseJobState):
    """StoppingState class."""

    def stop(self):
        """Transition to STOPPING state."""
        print("Stopping job...")
        self.job.set_state(JobStateEnum.STOPPING)

    def cond_updated(self):
        """Handle the transition to running."""
        self.job.set_state(JobStateEnum.RUNNING)


class JobStateContext:
    """JobStateContext class."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.state = ReadyState(self)
        self.state_enum = JobStateEnum.READY

    def set_state(self, state_enum):
        """Transition the job to a new state."""
        self.state_enum = state_enum
        self.state = self._get_state_class(state_enum)()

    def _get_state_class(self, state_enum):
        """Map a JobStateEnum to its corresponding state class."""
        state_mapping = {
            JobStateEnum.READY: ReadyState,
            JobStateEnum.RUNNING: RunningState,
            JobStateEnum.STARTING: StartingState,
            JobStateEnum.STOPPED: StoppedState,
            JobStateEnum.STOPPING: StoppingState,
            JobStateEnum.UPDATING: UpdatingState,
        }
        return state_mapping[state_enum](self)

    def start(self):
        """Transition to STARTING state."""
        self.state.start()

    def stop(self):
        """Transition to STOPPING state."""
        self.state.stop()

    def update(self):
        """Transition to UPDATING state."""
        self.state.update()

    def cond_running(self):
        """Handle the transition to running."""
        self.state.cond_running()

    def cond_updated(self):
        """Handle the transition to running."""
        self.state.cond_updated()

    def cond_stopped(self):
        """Handle the transition to stopped."""
        self.state.cond_stopped()
