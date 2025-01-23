from abc import ABC, abstractmethod
from enum import Enum

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

    def split(self, job_config: JobConfig):
        """Split the job config using even deployment policy."""
        # Placeholder method for splitting job config
        pass


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def split(self, job_config: JobConfig):
        """Split the job config using random deployment policy."""
        # Placeholder method for splitting job config
        pass


class DeploymentPolicyFactory:
    """Deployment policy factory class."""

    def get_deployment(self, deployment_policy: DeploymentPolicyEnum) -> DeploymentPolicy:
        """Return deployment policy class instance."""
        policies = {
            DeploymentPolicyEnum.RANDOM: RandomDeploymentPolicy(),
            DeploymentPolicyEnum.EVEN: EvenDeploymentPolicy(),
        }

        return policies[deployment_policy]
