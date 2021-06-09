# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tegrastats profiler class to measure CPU and GPU performance of benchmark tests."""

from datetime import datetime
import os
import re
import subprocess
from typing import Dict

import numpy as np

from .profiler import Profiler


class TegrastatsProfiler(Profiler):
    """Tegrastats profiler class to measure CPU and GPU performance of benchmark tests."""

    def __init__(self, command_path: str = '/tegrastats'):
        """
        Construct Tegrastats profiler.

        Parameters
        ----------
        command_path : str
            Path to invoke the profiler binary

        """
        self.tegrastats_path = command_path
        self.profiler_running = False

    def start_profiling(self, interval: float, log_dir: str) -> str:
        """
        Run tegrastats profiling program to keep track of performance metrics.

        Parameters
        ----------
        interval : float
            Interval for the profiler
        log_dir : str
            Path to write the logs to

        Returns
        -------
        str
            The path to the file where the profiler will write its logs to. If start_profiling()
            is called when the profiler is still running, return an empty string

        """
        if not self.profiler_running:
            file_name = str(datetime.timestamp(datetime.now())) + '.txt'
            file_path = os.path.join(log_dir, file_name)
            subprocess.Popen([self.tegrastats_path, '--interval',
                              str(interval), '--logfile', file_path])
            self.profiler_running = True
            return file_path
        else:
            print('Profiler is already running.')
            return ''

    def stop_profiling(self):
        """Stop profiling after running start_profiling()."""
        if self.profiler_running:
            subprocess.Popen([self.tegrastats_path, '--stop'])
            self.profiler_running = False

    def print_profiling_results(self, logfile_path: str) -> Dict[str, float]:
        """
        Parse Tegrastats profiling results from the logs and print them.

        This should only be called after calling start_profiling() and stop_profiling().

        Parameters
        ----------
        logfile_path : str
            The path to the logfile that will be parsed

        Returns
        -------
        Dict[str, float]
            Dictionary where each key is the metric name, and the value is the metric's value

        """
        data = {}
        gpu_values = []
        cpu_values = []
        with open(logfile_path) as file:
            for line in file.readlines():
                fields = line.split()
                GPU = fields[13]
                CPU = fields[9][1:-1]
                gpu_values.append(float(GPU[:-1]))
                cpu_array = re.split('%@[0-9]+[,]?', CPU)
                cpu_array = [float(value) for value in cpu_array[:-1]]
                cpu_values.append(np.mean(cpu_array))
            data['gpu_mean'] = np.mean(gpu_values)
            data['gpu_deviation'] = np.std(gpu_values)
            data['gpu_max'] = max(gpu_values)
            data['gpu_min'] = min(gpu_values)
            data['cpu_mean'] = np.mean(cpu_values)
            data['cpu_dev'] = np.std(cpu_values)
            data['cpu_max'] = max(cpu_values)
            data['cpu_min'] = min(cpu_values)
        return data
