# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
