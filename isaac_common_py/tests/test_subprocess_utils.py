# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import pathlib
import shutil

from isaac_common_py import subprocess_utils


def test_run_command(tmp_path: pathlib.Path):
    output = subprocess_utils.run_command(
        mnemonic='Example Command',
        command='ping google.com -c 10'.split(),
        log_file=tmp_path / 'log.txt',
        print_mode='tail',
    )
    assert len(output) > 10
