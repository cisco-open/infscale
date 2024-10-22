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

"""constants."""

# This file contains project-level constants.
# Note: DO NOT ADD CONSTANTS SPECIFIC TO A SINGLE FILE OR SUBMODULE

APISERVER_PORT = 8080
APISERVER_ENDPOINT = f"http://localhost:{APISERVER_PORT}"
CONTROLLER_PORT = 31310
GRPC_MAX_MESSAGE_LENGTH = 1073741824  # 1GB
HEART_BEAT_PERIOD = 3  # 3 seconds; heart beat between controller and agent
LOCALHOST = "127.0.0.1"
