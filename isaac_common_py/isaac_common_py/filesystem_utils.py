# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import os


def create_workdir(base_path: pathlib.Path, version: str, allow_sudo=False) -> pathlib.Path:
    """ Create a versioned workdir with a latest symlink. """
    work_path = base_path / version
    try:
        work_path.mkdir(exist_ok=True, parents=True)
    except PermissionError as e:
        if allow_sudo:
            os.system(f'sudo mkdir -p {base_path}')
        else:
            raise e

    if not os.access(base_path, os.W_OK):
        if allow_sudo:
            os.system(f'sudo chown {os.getuid()} {base_path}')
        # If sudo is not allowed we don't raise an error here, since we expect
        # one of the commands below to raise the correct PermissionError.

    latest_work_path = base_path / "latest"
    latest_work_path.unlink(missing_ok=True)
    latest_work_path.symlink_to(work_path, target_is_directory=True)
    return work_path
