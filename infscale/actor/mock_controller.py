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

import asyncio

import yaml
from infscale.config import JobConfig


class MockController:
    """JobManager class."""

    def __init__(self, file_paths: list[str]):
        self.config_q = asyncio.Queue()
        self.file_paths = file_paths

    async def start_sending(self):
        for file in self.file_paths:
            with open(file) as f:
                spec = yaml.safe_load(f)
                job_config = JobConfig(**spec)
                await self.config_q.put(job_config)
            
            await asyncio.sleep(20)