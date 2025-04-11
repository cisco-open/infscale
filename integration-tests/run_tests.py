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

import os
import subprocess
import sys
import tempfile

import yaml
from tests_dtype import TestConfig


def run_tests(config: str):
    """Run tests based on config."""
    with open(config) as f:
        config = yaml.safe_load(f)

    for item in config:
        cfg = str(TestConfig(**item))
        _run(cfg)


def _run(test_content: str) -> None:
    """Run single test using config."""
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yml"
    ) as temp_file:
        temp_file.write(test_content)
        temp_file.close()

        command = [
            "ansible-playbook",
            "-i",
            "inventory.yaml",
            temp_file.name,
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # re-route output to the terminal for visual feedback
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            print(f"\n {temp_file.name} failed with exit code {process.returncode}")
        else:
            print(f"\n {temp_file.name} completed successfully.")

        os.remove(temp_file.name)


if __name__ == "__main__":
    run_tests(sys.argv[1])
