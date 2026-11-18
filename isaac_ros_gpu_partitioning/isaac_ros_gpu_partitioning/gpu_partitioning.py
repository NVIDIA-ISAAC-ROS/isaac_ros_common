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

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import pwd
import subprocess
import time

from launch.actions import SetEnvironmentVariable
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetUUID, nvmlInit, nvmlShutdown
from pynvml import NVMLError
import yaml


@dataclass
class gpuPartitionConfig:
    name: str
    device: str
    chunks: int


@dataclass
class gpuPartition:
    name: str
    device: str
    chunks: int
    sm_count: int
    partition_id: int


class gpuServerState(Enum):
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERROR = 'ERROR'
    CONFIGURED = 'CONFIGURED'


CUDA_MPS_DIRECTORY = '/tmp/nvidia-mps'
GPU_PARTITION_FILE = CUDA_MPS_DIRECTORY + '/gpu_partitions.yaml'


def _mps_pipe_directory(pid):
    """Return the CUDA MPS pipe directory of process pid, or None if it cannot be read."""
    try:
        with open(f'/proc/{int(pid)}/environ', 'rb') as environ:
            variables = environ.read().split(b'\0')
    except OSError:
        return None
    for variable in variables:
        name, _, value = variable.partition(b'=')
        if name == b'CUDA_MPS_PIPE_DIRECTORY':
            return os.path.normpath(value.decode(errors='replace'))
    # CUDA MPS also puts its pipe in CUDA_MPS_DIRECTORY when the variable is unset.
    return CUDA_MPS_DIRECTORY


def _running_mps_processes():
    """Return the command lines of the CUDA MPS daemon and servers on CUDA_MPS_DIRECTORY."""
    result = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True,
                            check=False)
    processes = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 1)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        args = command.split()
        name = os.path.basename(args[0])
        is_mps = name == 'nvidia-cuda-mps-server' or (
            name == 'nvidia-cuda-mps-control' and '-d' in args)
        # A daemon on another pipe belongs to another workload. quit through
        # CUDA_MPS_DIRECTORY cannot reach it, so it is neither stopped nor waited for.
        # One whose environment cannot be read, such as a server started for another
        # user, still counts: skipping it could leave a leaked daemon running.
        if is_mps and _mps_pipe_directory(pid) in (CUDA_MPS_DIRECTORY, None):
            processes.append(command.strip())
    return processes


def stop_cuda_mps_daemon(timeout_sec=60.0):
    """
    Quit the CUDA MPS control daemon on CUDA_MPS_DIRECTORY and wait for it to exit.

    Unlike GpuPartition.stop_cuda_mps_server(), this also quits a daemon that has
    partitions configured. quit waits for every connected MPS client to finish, so
    call this only after the processes launched into a partition have exited.
    MPS daemons on other pipe directories are left alone.
    """
    if not _running_mps_processes():
        return
    env = dict(os.environ,
               CUDA_MPS_PIPE_DIRECTORY=CUDA_MPS_DIRECTORY,
               CUDA_MPS_LOG_DIRECTORY=CUDA_MPS_DIRECTORY)
    try:
        subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', env=env,
                       capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f'CUDA MPS control daemon did not quit within {timeout_sec}s; '
            'an MPS client is probably still running.') from error
    # The wait for the exit gets its own budget, so a slow quit cannot use it up.
    deadline = time.monotonic() + timeout_sec
    while _running_mps_processes():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f'CUDA MPS processes still running {timeout_sec}s after quit: '
                f'{_running_mps_processes()}')
        time.sleep(0.5)
    if os.path.exists(GPU_PARTITION_FILE):
        os.remove(GPU_PARTITION_FILE)


class GpuPartition:
    """A class to manage GPU partitioning through Nvidia CUDA MPS."""

    def __init__(self, config_file=None, gpu_partitions=None, start_server=False,
                 clear_partitions=False, mps_directory=CUDA_MPS_DIRECTORY, mps_user=None):
        super().__init__()

        self._is_initialized = False
        self._gpu_partitions = list(gpu_partitions) if gpu_partitions else []
        self._gpu_partitions_file = GPU_PARTITION_FILE
        self._sm_count_per_trunk = 4
        self._default_trunks = 0
        self._total_sm_count = 0

        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = mps_directory
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = mps_directory

        if mps_user is None:
            uid = os.getuid()
            gid = os.getgid()
        else:
            uid = pwd.getpwnam(mps_user).pw_uid
            gid = pwd.getpwnam(mps_user).pw_gid
        if not os.path.exists(mps_directory):
            os.makedirs(mps_directory, exist_ok=True, mode=0o777)
            os.chown(mps_directory, uid, gid)
        else:
            print(f'CUDA MPS directory already exists: {mps_directory}')

        if not self._is_initialized:
            # Check MPS server running state
            state = self.get_gpu_server_state()

            if isinstance(state, gpuServerState) and state is gpuServerState.CONFIGURED:
                # Populate the GPU partitions from an internal file
                if clear_partitions:
                    self.clear_gpu_partitions()
                else:
                    self.gpu_partitions_from_file()
                    self._is_initialized = True
                    return
            if not config_file:
                if not self._gpu_partitions:
                    raise ValueError('Configured file path is not provided')
                for partition in self._gpu_partitions:
                    partition.sm_count = 0
                    partition.partition_id = 0

            else:
                self._gpu_partitions = self.read_partition_configuration(config_file)
        else:
            print('CUDA MPS static partition is already intialized.')
            return

        # start the CUDA MPS server
        if start_server:
            state = self.get_gpu_server_state()
            if state is gpuServerState.STOPPED:
                if not self.start_cuda_mps_server(uid=uid):
                    raise ValueError('Failed to start the CUDA MPS server.')

        print('Configuring CUDA MPS static partition...')
        state = self.get_gpu_server_state()
        if state is gpuServerState.RUNNING or state is gpuServerState.CONFIGURED:
            self.create_gpu_partition()
            self._is_initialized = True
        else:
            raise ValueError('Error: MPS server is not running for CUDA MPS static partition.')

    def get_sm_count_per_trunk(self):
        """Get the SM count per trunk."""
        if not self._is_initialized:
            return None

        return self._sm_count_per_trunk

    def clear_gpu_partitions(self):
        """Clear the GPU partitions."""
        # get all the partitions and remove them
        cmd = 'nvidia-cuda-mps-control "lspart"'
        stdout, stderr = self.shell_command(cmd)
        if stderr:
            raise ValueError(f'Clearing CUDA MPS Partitions Error: {stderr}')

        lines = stdout.split('\n')
        for line in lines[3:]:
            if line.startswith('GPU-'):
                partition_id = line.split()[1]
                gpu_device = line.split()[0]
                cmd_options = f'sm_partition rm {gpu_device} {partition_id}'
                cmd = f'nvidia-cuda-mps-control "{cmd_options}"'
                stdout, stderr = self.shell_command(cmd)
                if stderr:
                    raise ValueError(f'Clearing CUDA MPS Partitions Error: {stderr}')
        print('CUDA MPS partitions cleared successfully.')

    def get_total_trunks(self):
        """Get the total SM count."""
        if not self._is_initialized:
            return None

        return self._default_trunks

    def get_total_sm_count(self):
        """Get the total SM count."""
        return self._total_sm_count

    def gpu_partitions_from_file(self):
        """Read the GPU partitions from a yaml file."""
        print(f'GPU Partitions File: {self._gpu_partitions_file}')

        with open(self._gpu_partitions_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)['sm_partitions']
            for item in data:
                chunks = int(item['chunks'])
                partition_id = str(item['partition_id'])
                sm_count = int(str(item['sm_count']))
                self._gpu_partitions.append(gpuPartition(name=str(item['name']),
                                                         device=str(item['device']),
                                                         chunks=chunks, sm_count=sm_count,
                                                         partition_id=partition_id))

    def read_partition_configuration(self, config_file):
        """Read the CUDA MPS partition."""
        with open(config_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)['sm_partitions']

        index = 0
        gpu_partitions = []
        for item in data:
            chunks = int(item['chunks'])
            gpu_partitions.append(gpuPartition(**item, partition_id=index, sm_count=chunks * 4))
            index += 1
        return gpu_partitions

    def shell_command(self, command):
        """Execute a shell command and capture stdout/stderr."""
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=False)
        stdout = result.stdout
        stderr = result.stderr
        if result.returncode != 0 and not stderr:
            stderr = f'Command failed with exit code {result.returncode}: {command}\n'
        if stderr:
            print(f'Error:\n {stderr}')
        return stdout, stderr

    def get_gpu_server_state(self) -> gpuServerState:
        """Get the GPU server state."""
        result, stderr = self.shell_command('nvidia-cuda-mps-control lspart')
        # Handle case that 'lspart' has no output at all
        if not result or result == '':
            daemon_running = False
            ps_result, _ = self.shell_command(
                'ps -ef | grep nvidia-cuda-mps-control')
            for line in ps_result.split('\n'):
                if 'nvidia-cuda-mps-control -d -s' in line:
                    daemon_running = True
            if not daemon_running:
                return gpuServerState.STOPPED
        # MPS server is running, check if configured
        lines = result.split('\n')
        if len(lines) > 3 and lines[3].startswith('GPU-'):
            return gpuServerState.CONFIGURED
        else:
            return gpuServerState.RUNNING

    def start_cuda_mps_server(self, uid=0):
        """Start the CUDA MPS server."""
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = CUDA_MPS_DIRECTORY
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = CUDA_MPS_DIRECTORY

        # Find the CUDA MPS server version
        version_output, stderr = self.shell_command('nvidia-cuda-mps-control -v')
        version_prefix = 'Binary version: '
        version_parts = version_output.split(version_prefix, maxsplit=1)
        if len(version_parts) != 2:
            command_output = stderr.strip() or version_output.strip() or 'no output'
            raise ValueError(
                'Unable to determine CUDA MPS server version: '
                f'nvidia-cuda-mps-control -v returned {command_output}')
        version = version_parts[1].split()[0] if version_parts[1].split() else ''
        try:
            version_number = int(version)
        except ValueError as error:
            raise ValueError(f'Invalid CUDA MPS server version: {version}') from error
        print(f'CUDA MPS Server Version: {version}')
        if version_number < 13010:
            raise ValueError('Please update to version 13010 or higher to use GPU partitioning.')

        try:
            cmd = 'nvidia-cuda-mps-control -d -s -q'
            stdout, stderr = self.shell_command(cmd)
            if stderr:
                print(f'Error:\n {stderr}')
        except FileNotFoundError as error:
            print(f'Command not found: {error}')

        # Start sever explicitly
        print(f'Starting server with UID: {uid}')
        result, stderr = self.shell_command(
            f'echo start_server -uid {uid} | nvidia-cuda-mps-control')
        if (stderr):
            print(f'Error starting server: {stderr}')
            return False

        # Check mps daemon is running
        result, stderr = self.shell_command('ps -ef | grep nvidia-cuda-mps-control')
        for line in result.split('\n'):
            if 'nvidia-cuda-mps-control -d -s' in line:
                pid = line.split()[1]
                if not pid:
                    print('MPS control daemon PID not found.')
                    return False
                return True
        return False

    def stop_cuda_mps_server(self):
        """Stop the CUDA MPS server."""
        if not self._is_initialized:
            print('CUDA MPS server is not initialized. Skipping stop.')
            return

        if self.get_gpu_server_state() is not gpuServerState.RUNNING:
            print('CUDA MPS server is not running. Skipping stop.')
            return

        print('Stopping CUDA MPS server...')
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = CUDA_MPS_DIRECTORY
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = CUDA_MPS_DIRECTORY
        if os.path.exists(self._gpu_partitions_file):
            os.remove(self._gpu_partitions_file)
        _, stderr = self.shell_command('echo quit | nvidia-cuda-mps-control')
        if stderr:
            raise ValueError(f'Failed to stop CUDA MPS server: {stderr}')
        self._is_initialized = False
        return True

    @staticmethod
    def is_jetson_orin() -> bool:
        """Return whether the host uses an NVIDIA Jetson Orin SoC."""
        compatible = Path('/proc/device-tree/compatible')
        try:
            identifiers = compatible.read_bytes().split(b'\0')
        except OSError:
            return False
        return b'nvidia,tegra234' in identifiers

    def create_gpu_partition(self):
        # Get the CUDA MPS server partitions
        cmd = 'nvidia-cuda-mps-control "lspart"'
        stdout, stderr = self.shell_command(cmd)

        # get the SM count per trunk
        lines = stdout.split('\n')
        if len(lines) > 4:
            raise ValueError('Multiple GPU partitions found. Please check the GPU partitions'
                             'configuration.')
        if len(lines) < 3:
            raise ValueError(f'Unexpected lspart output: {stdout!r}')
        lspart_uuid = lines[2].split()[0]
        if lines[2].startswith(str(lspart_uuid)):
            free_chunks = int(lines[2].split()[1])
            free_SM = int(lines[2].split()[3])
            self._sm_count_per_trunk = int(free_SM/free_chunks)
            self._default_trunks = free_chunks
            self._total_sm_count = free_SM

        device_uuid = lspart_uuid
        if not self.is_jetson_orin():
            gpu_id = self._gpu_partitions[0].device
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(int(gpu_id))
                device_uuid = nvmlDeviceGetUUID(handle)[:12]
                print(f'GPU {gpu_id} UUID: {device_uuid}')
            except NVMLError as error:
                raise ValueError(
                    f'Failed to identify GPU {gpu_id} with NVML: {error}') from error
            finally:
                try:
                    nvmlShutdown()
                except NVMLError as error:
                    print(f'Error: {error}')

        add_default = False
        for partition in self._gpu_partitions[:]:
            # only add non-default partitions
            if (partition.name == 'default' or partition.name == 'DEFAULT') and \
               partition.chunks == 0:  # noqa: E501
                # add default at last position
                add_default = True
                default_partition = partition
                self._gpu_partitions.remove(partition)
            else:
                cmd_options = f'sm_partition add {device_uuid} {partition.chunks}'
                cmd = f'nvidia-cuda-mps-control "{cmd_options}"'
                stdout, stderr = self.shell_command(cmd)
                if stderr:
                    raise ValueError(f'Creating CUDA MPS Partition Error: {stderr}')
                if 'Failed' in stdout:
                    raise ValueError(f'Creating CUDA MPS Partition for {partition.chunks} chunks'
                                     f' failed, free chunks: {self._default_trunks}.'
                                     f' Error: {stdout}')
                output_tokens = stdout.split()
                if len(output_tokens) < 2 or '/' not in output_tokens[1]:
                    raise ValueError(f'Unexpected CUDA MPS partition output: {stdout!r}')
                partition_id = output_tokens[1].split('/', 1)[1]
                partition.partition_id = f'{lspart_uuid}/{partition_id}'
                partition.sm_count = partition.chunks * self._sm_count_per_trunk
                self._default_trunks -= partition.chunks

        if add_default and self._default_trunks > 0:
            # add default at last position
            cmd_options = f'sm_partition add {device_uuid} {self._default_trunks}'
            cmd = f'nvidia-cuda-mps-control "{cmd_options}"'
            stdout, stderr = self.shell_command(cmd)
            if stderr:
                raise ValueError(f'Creating CUDA MPS Partition Error: {stderr}')
            output_tokens = stdout.split()
            if len(output_tokens) < 2 or '/' not in output_tokens[1]:
                raise ValueError(f'Unexpected CUDA MPS partition output: {stdout!r}')
            partition_id = f'{lspart_uuid}/{output_tokens[1].split("/", 1)[1]}'
            default_partition.chunks = self._default_trunks
            default_partition.partition_id = partition_id
            default_partition.sm_count = self._default_trunks * self._sm_count_per_trunk
            self._gpu_partitions.append(default_partition)
        self.generate_gpu_partitions_file()

    def generate_gpu_partitions_file(self):
        """Generate the GPU partitions file."""
        with open(self._gpu_partitions_file, 'w') as f:
            f.write('%YAML 1.2\n')
            f.write('---\n')
            f.write('sm_partitions:\n')
            for partition in self._gpu_partitions:
                f.write(f'  - name: {partition.name}\n')
                f.write(f'    device: {partition.device}\n')
                f.write(f'    chunks: {partition.chunks}\n')
                f.write(f'    sm_count: {partition.sm_count}\n')
                f.write(f'    partition_id: {partition.partition_id}\n')
            f.close()

    def get_partition_id(self, name):
        """Get the partition ID for a given partition name."""
        if not self._is_initialized:
            print('CUDA MPS server is not initialized.')
            return None

        for partition in self._gpu_partitions:
            if partition.name == name:
                return partition.partition_id, partition.sm_count
        print(f'Partition ID not found for name: {name}')
        return None

    def gpu_partition_launch_action(self, name):
        """Generate the launch action."""
        if not self._is_initialized:
            return None

        for partition in self._gpu_partitions:
            if partition.name == name:
                action = SetEnvironmentVariable(name='CUDA_MPS_SM_PARTITION',
                                                value=f'{partition.partition_id}')
                return action
        return None

    def gpu_partition_launch_action_group(self, names, containers):
        """Generate the launch actions for a group containers."""
        if not self._is_initialized:
            return None

        actions = []
        for name, container in zip(names, containers):
            action = self.gpu_partition_launch_action(name)
            if action:
                actions.append(action)
                actions.append(container)
            else:
                raise ValueError(f'Partition ID not found for name: {name}')
        return actions
