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

import sys


def print_green(text: str):
    """ Print text in green. """
    print(f"\033[32m{text}\033[0m")


def print_yellow(text: str):
    """ Print text in yellow. """
    print(f"\033[33m{text}\033[0m")


def print_blue(text: str):
    """ Print text in blue. """
    print(f"\033[34m{text}\033[0m")


def print_gray(text: str):
    """ Print text in gray. """
    print(f"\033[90m{text}\033[0m")


def print_red(text: str):
    """ Print text in red. """
    print(f"\033[91m{text}\033[0m")


def delete_last_lines_in_stdout(n: int):
    """ Delete the last n lines in stdout. """
    sys.stdout.write("\033[F\033[K" * n)
