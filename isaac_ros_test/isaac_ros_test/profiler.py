# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Profiler base class to measure the performance of benchmark tests."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


class Profiler(ABC):
    """Profiler base class to measure the performance of benchmark tests."""

    @abstractmethod
    def __init__(self):
        """Construct profiler."""
        self.is_running = False

        # Logfile path is generated once start_profiling() is called
        self.logfile_path = None

    @abstractmethod
    def start_profiling(self, log_dir: Path) -> None:
        """
        Run profiling program to keep track of performance metrics.

        Parameters
        ----------
        log_dir : Path
            Path to write the logs to

        """
        assert not self.is_running, 'Profiler has already been started!'
        self.is_running = True

        # Create logfile parent folders if they don't exist already
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logfile_path = log_dir / \
            f'{type(self).__name__}_{datetime.timestamp(datetime.now())}.log'

        return self.logfile_path

    @abstractmethod
    def stop_profiling(self) -> None:
        """Stop profiling after running start_profiling()."""
        assert self.is_running, 'Profiler was not yet started!'
        self.is_running = False

    @abstractmethod
    def get_results(self, logfile_path: Path = None) -> Dict[str, Tuple[str, str]]:
        """
        Return a labelled dictionary of results parsed from the logfile.

        This should only be called after calling start_profiling() and stop_profiling().

        Returns
        -------
        Dict[str, str]
            Dictionary mapping label: (value, units)

        """
        return
