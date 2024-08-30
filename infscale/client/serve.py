# Copyright 2024 Cisco Systems, Inc. and its affiliates
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

"""implementation of serve restapi call."""

import asyncio

import yaml
from infscale.actor.worker import Worker
from infscale.config import ServeConfig
from infscale.openapi import ApiClient, Configuration, DefaultApi, ServeSpec


def serve(host: str, port: int, specfile: str):
    """Call serve restapi."""
    endpoint = f"http://{host}:{port}"

    with open(specfile, "r") as f:
        spec_dict = yaml.safe_load(f)

    config = Configuration(endpoint)
    config.client_side_validation = False
    with ApiClient(config) as api_client:
        # Create an instance of the API class
        api_instance = DefaultApi(api_client)
        var_self = None
        spec = ServeSpec(**spec_dict)

        try:
            api_response = api_instance.serve_models_post(var_self, spec)
            print(f"{api_response.message}")
            # pprint(api_response)
        except Exception as e:
            print(f"Exception during serve api call: {e}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(Worker(0, None, ServeConfig(**spec_dict))._run())
