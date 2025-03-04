# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""factory.py."""

from infscale.controller.deployment.even import EvenDeploymentPolicy
from infscale.controller.deployment.policy import (DeploymentPolicy,
                                                   DeploymentPolicyEnum)
from infscale.controller.deployment.random import RandomDeploymentPolicy
from infscale.controller.deployment.static import StaticDeploymentPolicy


class DeploymentPolicyFactory:
    """Deployment policy factory class."""

    def get_deployment(
        self, deployment_policy: DeploymentPolicyEnum
    ) -> DeploymentPolicy:
        """Return deployment policy class instance."""
        policies = {
            DeploymentPolicyEnum.RANDOM: RandomDeploymentPolicy(),
            DeploymentPolicyEnum.EVEN: EvenDeploymentPolicy(),
            DeploymentPolicyEnum.STATIC: StaticDeploymentPolicy(),
        }

        return policies[deployment_policy]
