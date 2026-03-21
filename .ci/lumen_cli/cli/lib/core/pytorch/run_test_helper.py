"""Thin wrappers around common test invocation patterns."""

from __future__ import annotations

import sys

from cli.lib.common.utils import run_command


def run_test(*args: str) -> None:
    """Invoke python test/run_test.py with the given arguments."""
    cmd = f"{sys.executable} test/run_test.py " + " ".join(args)
    run_command(cmd)


def run_command_checked(cmd: str) -> None:
    """Run an arbitrary shell command, raising on failure."""
    run_command(cmd, use_shell=True)
