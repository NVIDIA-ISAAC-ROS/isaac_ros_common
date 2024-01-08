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

"""CPU profiler class to measure performance of benchmark tests on x86."""

from pathlib import Path
from threading import Thread
from typing import Dict, Tuple

import numpy as np
import psutil

from .profiler import Profiler


class CPUProfiler(Profiler):
    """CPU profiler class to measure CPU performance of benchmark tests."""

    def __init__(self):
        """Construct CPU profiler."""
        super().__init__()

    def start_profiling(self, log_dir: Path, interval: float = 1.0) -> Path:
        """
        Start CPU profiling thread to keep track of performance metrics.

        Parameters
        ----------
        log_dir : Path
            Path to write the logs to

        interval: float
            The interval between measurements, in seconds

        """
        super().start_profiling(log_dir)

        # While the is_running flag is true, log CPU usage
        def psutil_log():
            with open(self.logfile_path, 'w+') as logfile:
                while self.is_running:
                    logfile.write(
                        f'{psutil.cpu_percent(interval=interval, percpu=True)}\n')

        self.psutil_thread = Thread(target=psutil_log)
        self.psutil_thread.start()

        return self.logfile_path

    def stop_profiling(self):
        """Stop profiling after running start_profiling()."""
        super().stop_profiling()

        # Now that the is_running flag has been set to false, wait for thread to stop
        self.psutil_thread.join()

    def get_results(self, logfile_path: Path = None) -> Dict[str, Tuple[str, str]]:
        """
        Return a labelled dictionary of results parsed from the logfile.

        This should only be called after calling start_profiling() and stop_profiling().

        Returns
        -------
        Dict[str, str]
            Dictionary mapping label: (value, units)

        """
        assert not self.is_running, 'Cannot collect results until profiler has been stopped!'

        logfile_path = self.logfile_path if logfile_path is None else logfile_path
        assert self.logfile_path is not None, 'No logfile to read results from!'

        data = {}
        with open(logfile_path) as logfile:
            cpu_values = []
            for line in logfile.readlines():
                # Remove brackets from line before splitting entries by comma
                cpu_values.append(np.mean([float(v)
                                  for v in line[1:-2].split(',')]))

            cpu_values = np.array(cpu_values)
            data['cpu_mean'] = np.mean(cpu_values)
            data['cpu_dev'] = np.std(cpu_values)
            data['cpu_min'] = np.min(cpu_values)
            data['cpu_max'] = np.max(cpu_values)
            data['cpu_baseline'] = cpu_values[0]

        return data
