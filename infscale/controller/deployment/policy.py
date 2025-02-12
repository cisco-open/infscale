from abc import ABC, abstractmethod
import copy
from enum import Enum

from infscale.config import JobConfig


class DeploymentPolicyEnum(Enum):
    """Deployment policy enum."""

    EVEN = "even"
    RANDOM = "random"


class DeploymentPolicy(ABC):
    """Abstract class for deployment policy."""

    @abstractmethod
    def split(self, job_config: JobConfig) -> dict[str, JobConfig]:
        """
        Split the job config using random deployment policy
        and return updated job config for each agent.
        """
        pass

    def _get_agent_updated_cfg(
        self, wrk_distr: dict[str, list[str]], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """Return updated job config for each agent."""
        agents_config = {}
        for agent_id, wrk_ids in wrk_distr.items():
            # create a job_config copy to update and pass it to the agent.
            cfg = copy.deepcopy(job_config)

            for w in cfg.workers:
                # set the deploy flag if the worker is in worker distribution for this agent
                w.deploy = w.id in wrk_ids

            agents_config[agent_id] = cfg

        return agents_config
