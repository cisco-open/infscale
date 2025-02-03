from abc import ABC, abstractmethod
import copy
from enum import Enum
import random

from infscale import get_logger
from infscale.config import JobConfig

logger = None


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


class EvenDeploymentPolicy(DeploymentPolicy):
    """Even deployment policy class."""

    def __init__(self):
        global logger
        logger = get_logger()

    def split(
        self, agent_ids: list[str], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """
        Split the job config using even deployment policy
        and return updated job config for each agent.

        Workers are distributed as evenly as possible across the available agents.
        If the number of workers isn't perfectly divisible by the number of agents,
        the "extra" workers are assigned to the first agents in the list.
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

            # assign only worker id to the current agent
            distribution[agent_id] = [
                worker.id
                for worker in workers[start_index : start_index + num_workers_for_agent]
            ]

            # move the start index to the next batch of workers
            start_index += num_workers_for_agent

        logger.info(f"got new worker distribution for agents: {distribution}")

        return self._get_agent_updated_cfg(distribution, job_config)


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def __init__(self):
        global logger
        logger = get_logger()

    def split(
        self, agent_ids: list[str], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """
        Split the job config using random deployment policy
        and return updated job config for each agent.

        Each agent gets at least one worker from the shuffled list.
        The remaining workers are distributed randomly.
        The random.shuffle(workers) ensures that the initial distribution
        of workers to agents is random.
        The random.choice(agent_ids) assigns the remaining workers in a random way,
        ensuring no agent is left out.
        """

        # make a copy of the workers list
        workers = job_config.workers[:]

        # start by assigning one worker to each agent randomly
        random.shuffle(workers)  # shuffle workers to ensure randomness
        distribution = {agent_id: [workers.pop().id] for agent_id in agent_ids}

        # distribute the remaining workers randomly
        while workers:
            agent_id = random.choice(agent_ids)  # choose an agent randomly
            distribution[agent_id].append(workers.pop().id)

        logger.info(f"got new worker distribution for agents: {distribution}")

        return self._get_agent_updated_cfg(distribution, job_config)


class DeploymentPolicyFactory:
    """Deployment policy factory class."""

    def get_deployment(
        self, deployment_policy: DeploymentPolicyEnum
    ) -> DeploymentPolicy:
        """Return deployment policy class instance."""
        policies = {
            DeploymentPolicyEnum.RANDOM: RandomDeploymentPolicy(),
            DeploymentPolicyEnum.EVEN: EvenDeploymentPolicy(),
        }

        return policies[deployment_policy]
