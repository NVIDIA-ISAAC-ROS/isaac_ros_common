# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse


def str_to_bool(value):
    """Convert string representation of boolean to actual boolean.

    Supports: true, false, True, False, 1, 0, yes, no, on, off
    Also treats --flag (without value) as true
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
        else:
            raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

    # Handle case where --flag is passed without a value (None)
    if value is None:
        return True

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {type(value)}")
