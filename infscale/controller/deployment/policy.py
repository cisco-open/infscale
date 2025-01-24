from abc import ABC, abstractmethod
from enum import Enum
import random

from infscale.config import JobConfig


class DeploymentPolicyEnum(Enum):
    """Deployment policy enum."""

    EVEN = "even"
    RANDOM = "random"


class DeploymentPolicy(ABC):
    """Abstract class for deployment policy."""

    @abstractmethod
    def split(self, job_config: JobConfig):
        """Split the job config using deployment policy."""
        # Placeholder method for splitting job config
        pass


class EvenDeploymentPolicy(DeploymentPolicy):
    """Even deployment policy class."""

    def split(self, agent_ids: list[str], job_config: JobConfig) -> dict[str, str]:
        """
        Split the job config using even deployment policy.

        Workers are distributed as evenly as possible across the available agents.
        If the number of workers isn't perfectly divisible by the number of agents,
        the "extra" workers are assigned to the first agents in the list
        """
        # dictionary to hold the workers for each agent_id
        distribution = {agent_id: [] for agent_id in agent_ids}

        num_agents = len(agent_ids)
        workers = job_config.workers

        # assign workers to agents evenly by splitting the list of workers
        workers_per_agent = len(workers) // num_agents
        remaining_workers = len(workers) % num_agents

        start_index = 0
        for i, agent_id in enumerate(agent_ids):
            # for the first 'remaining_workers' agents, assign one extra worker
            num_workers_for_agent = workers_per_agent + (
                1 if i < remaining_workers else 0
            )

            # assign only worker["id"] to the current agent
            distribution[agent_id] = [
                worker["id"]
                for worker in workers[start_index : start_index + num_workers_for_agent]
            ]

            # move the start index to the next batch of workers
            start_index += num_workers_for_agent

        return distribution


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def split(self, agent_ids: list[str], job_config: JobConfig):
        """
        Split the job config using random deployment policy.

        Each agent gets at least one worker from the shuffled list.
        The remaining workers are distributed randomly.
        The random.shuffle(workers) ensures that the initial distribution
        of workers to agents is random.
        The random.choice(agent_ids) assigns the remaining workers in a random way,
        ensuring no agent is left out
        """

        # make a copy of the workers list
        workers = job_config.workers[:]

        # start by assigning one worker to each agent randomly
        random.shuffle(workers)  # shuffle workers to ensure randomness
        distribution = {agent_id: [workers.pop()["id"]] for agent_id in agent_ids}

        # distribute the remaining workers randomly
        while workers:
            agent_id = random.choice(agent_ids)  # choose an agent randomly
            distribution[agent_id].append(workers.pop()["id"])

        return distribution


class DeploymentPolicyFactory:
    """Deployment policy factory class."""

    def get_deployment(self, deployment_policy: DeploymentPolicyEnum) -> DeploymentPolicy:
        """Return deployment policy class instance."""
        policies = {
            DeploymentPolicyEnum.RANDOM: RandomDeploymentPolicy(),
            DeploymentPolicyEnum.EVEN: EvenDeploymentPolicy(),
        }

        return policies[deployment_policy]
