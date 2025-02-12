from infscale import get_logger
from infscale.config import JobConfig
from infscale.controller.deployment.policy import DeploymentPolicy

logger = None


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
