import random
from infscale import get_logger
from infscale.config import JobConfig
from infscale.controller.deployment.policy import DeploymentPolicy

logger = None


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
