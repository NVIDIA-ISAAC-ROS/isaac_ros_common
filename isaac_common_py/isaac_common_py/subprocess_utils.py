import collections
import pathlib
import select
import subprocess
import time
from typing import Literal
from datetime import datetime, timedelta

from isaac_common_py import io_utils


def log_process_all(process: subprocess.Popen, log_file: pathlib.Path) -> list[str]:
    """ Log all output from the process to stdout and the log file. """
    full_output = []

    with log_file.open('w') as f:
        while process.poll() is None:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                output = process.stdout.readline().strip('\n')
                full_output.append(output)
                io_utils.print_gray(output)
                f.write(output + '\n')
                f.flush()

        stdout, _ = process.communicate()
        full_output.extend(stdout.splitlines())
        io_utils.print_gray(stdout)
        f.write(stdout)
        f.flush()

    process.wait()
    return full_output


def log_process_tail(process: subprocess.Popen, log_file: pathlib.Path, tail: int,
                     timeout=None) -> list[str]:
    """
    Log only the last n lines from the process output to stdout, but everything to the log file.
    """
    tail_output = collections.deque(maxlen=tail)
    full_output = []
    warning = ''
    start = time.time()
    with log_file.open('w') as f:
        while process.poll() is None:

            # Store the current output line.
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                # clear the previous warning
                if len(warning) > 0:
                    io_utils.delete_last_lines_in_stdout(1)
                    warning = ''

                io_utils.delete_last_lines_in_stdout(len(tail_output))

                output = process.stdout.readline().strip('\n')
                tail_output.append(output)
                full_output.append(output)

                # Print output and also add to log file.
                io_utils.print_gray('  ' + '\n  '.join(tail_output))
                f.write(output + '\n')
                f.flush()

            if timeout and time.time() - start > timeout:
                # clear previous warning
                if len(warning) > 0:
                    io_utils.delete_last_lines_in_stdout(1)

                warning = ('WARNING: The command has exceeded the timeout ' +
                           f'limit of {timeout} seconds.')
                io_utils.print_yellow(warning)

        stdout, _ = process.communicate()
        full_output.extend(stdout.splitlines())
        f.write(stdout)
        f.flush()

    process.wait()
    if len(warning) > 0:
        io_utils.delete_last_lines_in_stdout(1)
    io_utils.delete_last_lines_in_stdout(len(tail_output))
    return full_output


def log_process_none(process: subprocess.Popen, log_file: pathlib.Path) -> list[str]:
    """ Log nothing to stdout, but everything to the log file. """
    full_output = []
    with log_file.open('w') as f:
        while process.poll() is None:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                output = process.stdout.readline().strip('\n')
                full_output.append(output)
                f.write(output + '\n')
                f.flush()

        stdout, _ = process.communicate()
        f.write(stdout)
        f.flush()

    process.wait()
    return full_output


def run_command(
    mnemonic: str,
    command: str | list,
    log_file: pathlib.Path,
    print_mode: Literal['all', 'tail', 'none'],
    allow_failure=False,
    timeout=None,
    **kwargs,
) -> list[str]:
    """ Run a command and log its outputs. """
    assert print_mode in ['all', 'tail', 'none']
    if timeout:
        end = datetime.now() + timedelta(seconds=timeout)
        io_utils.print_blue(
            f'{mnemonic}: Estimated completion at {end.strftime("%H:%M:%S")}. Running...⏳')
    else:
        io_utils.print_blue(f'{mnemonic} Running...⏳')

    if not isinstance(command, list):
        command = [command]
    command = [str(c) for c in command]

    log_file.unlink(missing_ok=True)

    start = time.time()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )

    if print_mode == 'all':
        full_output = log_process_all(process, log_file)
    elif print_mode == 'tail':
        full_output = log_process_tail(process, log_file, 10, timeout)
    elif print_mode == 'none':
        full_output = log_process_none(process, log_file)

    end = time.time()
    duration = end - start

    success = True if allow_failure else process.returncode == 0
    status = (f'{mnemonic}: Success ✅️ [{duration:.2f}s]'
              if success else f'{mnemonic}: Fail ❌ [{duration:.2f}s]')

    if print_mode == 'all':
        io_utils.print_blue(status)
    elif print_mode in ['tail', 'none']:
        io_utils.delete_last_lines_in_stdout(1)
        io_utils.print_blue(status)
        if not success:
            io_utils.print_blue("Logs:")
            io_utils.print_gray('  ' + '\n  '.join(full_output))

    if not success:
        command_str = ' '.join(command)
        io_utils.print_red(f"Failed to run command '{command_str}'.")
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=[command_str])

    return full_output
