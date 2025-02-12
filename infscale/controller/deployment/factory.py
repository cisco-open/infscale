from infscale.controller.deployment.even import EvenDeploymentPolicy
from infscale.controller.deployment.policy import DeploymentPolicy, DeploymentPolicyEnum
from infscale.controller.deployment.random import RandomDeploymentPolicy


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
