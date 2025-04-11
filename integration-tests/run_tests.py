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

import subprocess


def run_tests():
    # start controller and agents
    _run("start_processes.yml")

    # start job
    _run("start_job.yml")

    # poll job status endpoint
    _run("poll_job_status.yml")

def _run(config_path: str) -> None:
    command = [
        "ansible-playbook",
        "-i", "inventory.yaml",
        config_path,
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
        print(f"\n {config_path} failed with exit code {process.returncode}")
    else:
        print(f"\n {config_path} completed successfully.")

if __name__ == "__main__":
    run_tests()
