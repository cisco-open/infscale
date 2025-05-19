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

"""run_tests.py."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TextIO

import pystache
import yaml
from constants import ERROR_PATTERN, LOG_FOLDER, PRINT_COLOR
from tests_dtype import Test, TestConfig


class IntegrationTest:
    """Integration test class."""

    def __init__(self, test_cfg_path: str, test_path: str):
        self.logs_sync_data: dict[str, subprocess.Popen] = {}
        self.local_log_path = os.path.join(LOG_FOLDER, "test.log")
        self.test_cfg: TestConfig = None
        self.test: Test = None
        self.inventory: str = ""
        self.is_failure = False
        self.monitor_logs_evt = threading.Event()
        self.log_thread: threading.Thread = None

        self._init(test_cfg_path, test_path)

    def _init(self, test_cfg_path, test_path):
        """Init test assets."""
        self._create_local_log()
        self._start_monitor_logs()

        with open(test_cfg_path) as f:
            test_cfg_file = yaml.safe_load(f)
            self.test_cfg = TestConfig(**test_cfg_file)

        self.inventory = self._generate_inventory(self.test_cfg)

        with open(test_path) as f:
            test_file = yaml.safe_load(f)
            self.test = Test(config=self.test_cfg, **test_file)

    def _create_local_log(self) -> None:
        """Create local log folder and file to store remote logs."""
        log_path = Path(f"{LOG_FOLDER}/test.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)

    def run_test(self):
        """Run test based on test file."""
        work_dir, steps = self.test.work_dir, self.test.steps
        ctrl_host = self.test_cfg.controller.host

        self._create_remote_logs_folder(self.test.work_dir)

        for step in steps:
            if self.is_failure:
                break

            self._start_remote_log_sync(step.host, step.work_dir)
            self._run_test(str(step))

        self._stop_remote_log_sync()
        self._cleanup(work_dir, ctrl_host)

    def _generate_inventory(self, test_cfg: TestConfig) -> str:
        """Generate inventory temp file based on hosts specified in config."""
        hosts = {hostname: {} for hostname in test_cfg.get_hosts()}

        inventory = {"all": {"hosts": hosts}}

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".yaml"
        ) as tmpfile:
            yaml.dump(inventory, tmpfile)
            tmpfile.flush()

            return tmpfile.name

    def _start_remote_log_sync(self, host: str, work_dir: str) -> None:
        """Launch a background process to get remote logs."""
        if host in self.logs_sync_data:
            return

        cmd = f"tail -f {LOG_FOLDER}/test.log | " f"stdbuf -oL sed 's/^/[{host}] /'"

        with open(self.local_log_path, "a") as f:
            p = subprocess.Popen(
                ["ssh", host, cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self.logs_sync_data[host] = p

    def _stop_remote_log_sync(self) -> None:
        """Terminate logs processes."""
        for proc in self.logs_sync_data.values():
            proc.terminate()

    def _create_remote_logs_folder(self, work_dir: str) -> None:
        """Create remote logs folder for each host.

        This folder will be used to store logs for controller and agents.
        Code will look for any python specific error to decide failure.
        """
        template = Path("templates/create_logs_folder.yaml").read_text()
        rendered = pystache.render(template, {"log_folder": LOG_FOLDER})

        file_name = self._get_temp_file(rendered)

        command = [
            "ansible-playbook",
            "-i",
            self.inventory,
            file_name,
        ]

        self._start_process(command, "create test logs folder")

    def _start_process(self, command: str, name: str) -> None:
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
            print(
                f"{PRINT_COLOR['failed']}\n'{name}' failed with exit code {process.returncode}.{PRINT_COLOR['black']}"
            )
            sys.exit(f"\n'{name}' failed with exit code {process.returncode}")
        else:
            print(
                f"{PRINT_COLOR['success']}\n'{name}' completed successfully.{PRINT_COLOR['black']}"
            )

    def _run_test(self, test_content: str) -> None:
        """Run single test using config."""
        file_name = self._get_temp_file(test_content)

        command = [
            "ansible-playbook",
            "-i",
            self.inventory,
            file_name,
        ]

        self._start_process(command, file_name)

        os.remove(file_name)

    def _cleanup(self, work_dir: str, ctrl_host: str) -> None:
        """Do cleanup after all tests are executed."""
        # remove temp local logs folder
        shutil.rmtree(LOG_FOLDER)

        template = Path("templates/cleanup_processes.yaml").read_text()
        rendered = pystache.render(
            template,
            {
                "work_dir": work_dir,
                "ctrl_host": ctrl_host,
                "log_folder": LOG_FOLDER,
            },
        )

        file_name = self._get_temp_file(rendered)

        command = [
            "ansible-playbook",
            "-i",
            self.inventory,
            file_name,
        ]

        self._start_process(command, "cleanup processes")
        self.monitor_logs_evt.set()

    def _get_temp_file(self, file: str) -> str:
        """Return a temporary yaml file name.

        This is a placeholder for ansible yaml file.
        This file will be used to run commands.
        """
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".yaml"
        ) as temp_file:
            temp_file.write(file)
            temp_file.close()

        return temp_file.name

    def _start_monitor_logs(self) -> None:
        """Start a new thread to monitor local logs file."""
        self.log_thread = threading.Thread(target=self._monitor_logs)
        self.log_thread.start()

    def _monitor_logs(self):
        """Continuously scan log file for multiline Traceback errors."""
        with open(self.local_log_path, "r") as f:
            f.seek(0, os.SEEK_END)  # Go to end of file

            while not self.monitor_logs_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue

                if ERROR_PATTERN.search(line):
                    self._handle_error_block(f, line)
                    self.monitor_logs_evt.set()
                    self.is_failure = True

    def _handle_error_block(self, file_obj: TextIO, first_line: str):
        """Collect and handle a multi-line error block starting with `first_line`."""
        buffer = [first_line]

        while True:
            next_line = file_obj.readline()
            if not next_line or next_line.strip() == "":
                break
            buffer.append(next_line)

        print(f"{PRINT_COLOR['failed']}Error:\n{''.join(buffer)}{PRINT_COLOR['black']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test using a test config.")
    parser.add_argument("--config", help="Path to test config file")
    parser.add_argument("--test", help="Path to test case file")

    args = parser.parse_args()

    int_test = IntegrationTest(args.config, args.test)
    int_test.run_test()
