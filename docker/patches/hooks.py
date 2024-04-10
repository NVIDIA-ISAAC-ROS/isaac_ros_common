# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pathlib
from unittest import TestCase

from _pytest._code.code import ReprFileLocation

import pytest

from ..loader import LoadTestsFromPythonModule
from ..test_runner import LaunchTestRunner


def _pytest_version_ge(major, minor=0, patch=0):
    """Return True if pytest version is >= the given version."""
    pytest_version = tuple([int(v) for v in pytest.__version__.split('.')])
    assert pytest_version
    return pytest_version >= (major, minor, patch)


class LaunchTestFailure(Exception):

    def __init__(self, message, results):
        super().__init__()
        self.message = message
        self.results = results

    def __str__(self):
        return self.message


class LaunchTestFailureRepr:
    """A `_pytest._code.code.ExceptionReprChain`-like object."""

    def __init__(self, failures):
        lines = [
            line
            for _, error_description in failures
            for line in error_description.splitlines()
        ]
        max_length = max(len(line) for line in lines) if lines else 3
        thick_sep_line = '=' * max_length
        thin_sep_line = '-' * max_length
        self._fulldescr = '\n' + '\n\n'.join([
            '\n'.join([
                thick_sep_line, f'FAIL: {test_name}',
                thin_sep_line, error_description
            ])
            for test_name, error_description in failures
        ])
        self.reprcrash = ReprFileLocation(
            path='', lineno=0, message='\n'.join([
                f'{test_name} failed' for test_name, _ in failures
            ])
        )

    def __str__(self):
        return self._fulldescr

    def toterminal(self, out):
        out.write(self._fulldescr)


class LaunchTestItem(pytest.Item):

    def __init__(self, parent, *, name):
        super().__init__(name, parent)
        self.test_runs = None
        self.runner_cls = None

    @classmethod
    def from_parent(
        cls, parent, *, name, test_runs, runner_cls=LaunchTestRunner
    ):
        """Override from_parent for compatibility."""
        # pytest.Item.from_parent didn't exist before pytest 5.4
        if hasattr(super(), 'from_parent'):
            instance = getattr(super(), 'from_parent')(parent, name=name)
        else:
            instance = cls(parent, name=name)
        instance.test_runs = test_runs
        instance.runner_cls = runner_cls
        return instance

    def runtest(self):
        launch_args = sum((
            args_set for args_set in self.config.getoption('--launch-args')
        ), [])
        # Copy test runs' collection as it may be used more than
        # once e.g. if pytest rerunfailures plugin is in use.
        test_runs = copy.deepcopy(self.test_runs)
        runner = self.runner_cls(
            test_runs=test_runs,
            launch_file_arguments=launch_args,
            debug=self.config.getoption('verbose')
        )

        runner.validate()
        results_per_run = runner.run()

        if any(not result.wasSuccessful() for result in results_per_run.values()):
            raise LaunchTestFailure(
                message='some test cases have failed', results=results_per_run
            )

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, LaunchTestFailure):
            return LaunchTestFailureRepr(failures=[
                (test_case.id(), formatted_error)
                for test_run, test_result in excinfo.value.results.items()
                for test_case, formatted_error in (test_result.errors + test_result.failures)
                if isinstance(test_case, TestCase) and not test_result.wasSuccessful()
            ])
        return super().repr_failure(excinfo)

    def reportinfo(self):
        if _pytest_version_ge(7):
            path = self.path
        else:
            path = self.fspath
        return path, 0, 'launch tests: {}'.format(self.name)


class LaunchTestModule(pytest.File):

    def __init__(self, *args, **kwargs):
        if _pytest_version_ge(7):
            if 'fspath' in kwargs:
                if kwargs['fspath'] is not None:
                    kwargs['path'] = pathlib.Path(kwargs['fspath'])
                del kwargs['fspath']
        super().__init__(*args, **kwargs)

    @classmethod
    def from_parent(cls, *args, **kwargs):
        """Override from_parent for compatibility."""
        # pytest.File.from_parent didn't exist before pytest 5.4
        if _pytest_version_ge(5, 4):
            return super().from_parent(*args, **kwargs)
        args_without_parent = args[1:]
        return cls(*args_without_parent, **kwargs)

    def makeitem(self, *args, **kwargs):
        return LaunchTestItem.from_parent(*args, **kwargs)

    def collect(self):
        if _pytest_version_ge(7):
            # self.path exists since 7
            from _pytest.pathlib import import_path
            module = import_path(self.path, root=None)
        else:
            module = self.fspath.pyimport()
        yield self.makeitem(
            name=module.__name__, parent=self,
            test_runs=LoadTestsFromPythonModule(
                module, name=module.__name__
            )
        )


def find_launch_test_entrypoint(path):
    try:
        if _pytest_version_ge(7):
            from _pytest.pathlib import import_path
            module = import_path(path, root=None, consider_namespace_packages=False)
        else:
            # Assume we got legacy path in earlier versions of pytest
            module = path.pyimport()
        return getattr(module, 'generate_test_description', None)
    except SyntaxError:
        return None


def pytest_pycollect_makemodule(path, parent):
    entrypoint = find_launch_test_entrypoint(path)
    if entrypoint is not None:
        ihook = parent.session.gethookproxy(path)
        module = ihook.pytest_launch_collect_makemodule(
            path=path, parent=parent, entrypoint=entrypoint
        )
        if module is not None:
            return module

    if _pytest_version_ge(7):
        path = pathlib.Path(path)
        if path.name == '__init__.py':
            return pytest.Package.from_parent(parent, path=path)
        return pytest.Module.from_parent(parent=parent, path=path)
    elif _pytest_version_ge(5, 4):
        if path.basename == '__init__.py':
            return pytest.Package.from_parent(parent, fspath=path)
        return pytest.Module.from_parent(parent, fspath=path)
    else:
        # todo: remove fallback once all platforms use pytest >=5.4
        if path.basename == '__init__.py':
            return pytest.Package(path, parent)
        return pytest.Module(path, parent)


@pytest.hookimpl(trylast=True)
def pytest_launch_collect_makemodule(path, parent, entrypoint):
    marks = getattr(entrypoint, 'pytestmark', [])
    if marks and any(m.name == 'launch_test' for m in marks):
        if _pytest_version_ge(7):
            path = pathlib.Path(path)
            module = LaunchTestModule.from_parent(parent=parent, path=path)
        else:
            module = LaunchTestModule.from_parent(parent=parent, fspath=path)
        for mark in marks:
            decorator = getattr(pytest.mark, mark.name)
            decorator = decorator.with_args(*mark.args, **mark.kwargs)
            module.add_marker(decorator)
        return module


def pytest_addhooks(pluginmanager):
    from launch_testing.pytest import hookspecs
    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser):
    parser.addoption(
        '--launch-args', action='append', nargs='*',
        default=[], help='One or more Launch test arguments'
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'launch_test: mark a generate_test_description function as a launch test entrypoint'
    )