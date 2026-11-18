#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import unittest
from unittest import mock

from isaac_ros_gpu_partitioning import gpu_partitioning

DAEMON = 'nvidia-cuda-mps-control -d -s -q'


class FakeClock:
    """Stands in for time.monotonic() and time.sleep(), so waits take no real time."""

    def __init__(self):
        """Start the clock at zero."""
        self.now = 0.0

    def monotonic(self):
        """Return the fake time in seconds."""
        return self.now

    def sleep(self, seconds):
        """Advance the fake time instead of waiting."""
        self.now += seconds


class TestStopCudaMpsDaemon(unittest.TestCase):
    """stop_cuda_mps_daemon() must leave no MPS daemon behind, or fail saying why."""

    def stop(self, snapshots, quit_seconds=0, quit_error=None):
        """Run stop_cuda_mps_daemon() against successive `ps` snapshots; return the quit mock."""
        snapshots = list(snapshots)
        clock = FakeClock()

        def running():
            """Return the next `ps` snapshot; the last one repeats."""
            return snapshots.pop(0) if len(snapshots) > 1 else snapshots[0]

        def quit_daemon(*args, **kwargs):
            """Stand in for the quit command: raise quit_error or take quit_seconds."""
            if quit_error:
                raise quit_error
            clock.sleep(quit_seconds)

        # os is mocked with the rest: a successful stop ends in
        # os.remove(GPU_PARTITION_FILE), and that path is machine-global
        # (/tmp/nvidia-mps/gpu_partitions.yaml), not per-test. Left real, this
        # unit test deletes the live partition manifest that GpuPartition
        # reads to recover its CONFIGURED state.
        with mock.patch.object(gpu_partitioning, '_running_mps_processes',
                               side_effect=running), \
                mock.patch.object(gpu_partitioning, 'os') as os_mock, \
                mock.patch.object(gpu_partitioning.subprocess, 'run',
                                  side_effect=quit_daemon) as quit_mock, \
                mock.patch.object(gpu_partitioning.time, 'sleep', side_effect=clock.sleep), \
                mock.patch.object(gpu_partitioning.time, 'monotonic',
                                  side_effect=clock.monotonic):
            os_mock.path.exists.return_value = False
            gpu_partitioning.stop_cuda_mps_daemon(timeout_sec=60)
        return quit_mock

    def test_does_nothing_without_daemon(self):
        """Send no quit when no daemon is running."""
        self.assertFalse(self.stop([[]]).called)

    def test_quits_daemon_and_waits_for_exit(self):
        """Send quit once, then wait until the daemon is gone."""
        quit_mock = self.stop([[DAEMON], [DAEMON], []])
        quit_mock.assert_called_once()
        self.assertEqual(quit_mock.call_args.kwargs['input'], 'quit\n')

    def test_slow_quit_leaves_the_full_wait_for_exit(self):
        """Keep the full wait for exit even when quit itself is slow."""
        # quit uses 59 of the 60 s, then the daemon takes 2.5 s more to exit.
        self.stop([[DAEMON]] * 6 + [[]], quit_seconds=59)

    def test_fails_when_quit_blocks(self):
        """Raise, rather than hang, when quit never returns."""
        blocked = subprocess.TimeoutExpired('nvidia-cuda-mps-control', 60)
        with self.assertRaises(RuntimeError):
            self.stop([[DAEMON]], quit_error=blocked)

    def test_fails_when_daemon_never_exits(self):
        """Raise when the daemon is still running after the timeout."""
        with self.assertRaises(RuntimeError):
            self.stop([[DAEMON]])


class TestRunningMpsProcesses(unittest.TestCase):
    """Only the MPS daemon and servers on CUDA_MPS_DIRECTORY count as running."""

    def test_skips_mps_processes_on_other_pipes(self):
        """Ignore daemons on other pipes; count ones whose pipe cannot be read."""
        ps = mock.Mock(stdout=(
            '   10 nvidia-cuda-mps-control -d -s -q\n'
            '   11 nvidia-cuda-mps-server\n'
            '   12 nvidia-cuda-mps-control -d\n'
            '   13 /usr/bin/nvidia-cuda-mps-server\n'
            '   14 python3 -m pytest\n'
            'PID nvidia-cuda-mps-server\n'))
        # 13's environment cannot be read, so it might be ours and still counts.
        pipes = {10: '/tmp/nvidia-mps', 11: '/tmp/nvidia-mps', 12: '/tmp/other-mps', 13: None}
        with mock.patch.object(gpu_partitioning.subprocess, 'run', return_value=ps), \
                mock.patch.object(gpu_partitioning, '_mps_pipe_directory',
                                  side_effect=pipes.get):
            self.assertEqual(gpu_partitioning._running_mps_processes(),
                             ['nvidia-cuda-mps-control -d -s -q', 'nvidia-cuda-mps-server',
                              '/usr/bin/nvidia-cuda-mps-server'])

    def test_reads_the_pipe_directory_from_the_process_environment(self):
        """Read CUDA_MPS_PIPE_DIRECTORY from /proc/<pid>/environ and normalize it."""
        environ = b'HOME=/root\0CUDA_MPS_PIPE_DIRECTORY=/tmp/other-mps/\0'
        with mock.patch('builtins.open', mock.mock_open(read_data=environ)):
            self.assertEqual(gpu_partitioning._mps_pipe_directory(12), '/tmp/other-mps')

    def test_unset_pipe_directory_means_the_default(self):
        """Fall back to the default pipe when the variable is unset."""
        with mock.patch('builtins.open', mock.mock_open(read_data=b'HOME=/root\0')):
            self.assertEqual(gpu_partitioning._mps_pipe_directory(10), '/tmp/nvidia-mps')

    def test_unreadable_environment_matches_no_pipe(self):
        """Return None when the environment cannot be read."""
        with mock.patch('builtins.open', side_effect=PermissionError):
            self.assertIsNone(gpu_partitioning._mps_pipe_directory(1))


if __name__ == '__main__':
    unittest.main()
