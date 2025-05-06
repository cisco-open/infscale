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

import argparse
import os
import subprocess
import tempfile

import yaml
from tests_dtype import Test, TestConfig


def run_tests(test_cfg_path: str, test_path: str):
    """Run tests based on config."""

    with open(test_cfg_path) as f:
        test_cfg_file = yaml.safe_load(f)
        test_cfg = TestConfig(**test_cfg_file)

    with open(test_path) as f:
        test_file = yaml.safe_load(f)
        test = Test(config=test_cfg, **test_file)

    for step in test.steps:
        _run(str(step))

    _cleanup()


def _run_process(command: str, name: str) -> None:
    """Run process with command."""
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
        print(f"\n {name} failed with exit code {process.returncode}")
    else:
        print(f"\n {name} completed successfully.")


def _run(test_content: str) -> None:
    """Run single test using config."""
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_content)
        temp_file.close()

        command = [
            "ansible-playbook",
            "-i",
            "inventory.yaml",
            temp_file.name,
        ]

        _run_process(command, temp_file.name)

        os.remove(temp_file.name)


def _cleanup() -> None:
    """Do cleanup after all tests are executed."""
    command = [
        "ansible-playbook",
        "-i",
        "inventory.yaml",
        "templates/cleanup_processes.yaml",
    ]

    _run_process(command, "cleanup processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test using a test config.")
    parser.add_argument("--config", help="Path to test config file")
    parser.add_argument("--test", help="Path to test case file")

    args = parser.parse_args()
    run_tests(args.config, args.test)
