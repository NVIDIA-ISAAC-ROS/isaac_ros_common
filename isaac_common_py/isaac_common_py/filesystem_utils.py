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
