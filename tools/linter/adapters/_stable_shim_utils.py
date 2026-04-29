"""
Shared utilities for stable-shim linters.

Consumed by:
    - tools/linter/adapters/stable_shim_version_linter.py
    - tools/linter/adapters/stable_shim_usage_linter.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.setup_helpers.gen_version_header import parse_version


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


class PreprocessorTracker:
    """
    Helper class to track preprocessor directives and version blocks.

    This class maintains state as it processes C/C++ preprocessor directives
    (#if, #elif, #else, #endif) and tracks which code is inside version blocks.
    """

    def __init__(self) -> None:
        # Stack of (is_version_block, version_tuple) tuples
        # is_version_block: True if this is a TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_0 block
        # version_tuple: (major, minor) if is_version_block is True, else None
        self.preprocessor_stack: list[tuple[bool, tuple[int, int] | None]] = []

        # Current version requirement (if inside a version block)
        self.version_of_block: tuple[int, int] | None = None

        # Track if we're inside a block comment
        self.in_block_comment: bool = False

        # Regex to match version conditions in #if or #elif
        self.version_pattern = re.compile(
            r"#(?:if|elif)\s+TORCH_FEATURE_VERSION\s*>=\s*TORCH_VERSION_(\d+)_(\d+)_\d+"
        )

    def process_line(self, line: str) -> bool:
        """
        Process a line and update the preprocessor state.

        Returns True if the line was a preprocessor directive or comment,
        False if it's a regular code line that should be further analyzed.
        """
        stripped = line.strip()

        # Handle block comments (/* ... */)
        if "/*" in line:
            self.in_block_comment = True

        if self.in_block_comment:
            if "*/" in line:
                self.in_block_comment = False
            return True

        # Skip line comments
        if stripped.startswith("//"):
            return True

        # Track #if directives
        if stripped.startswith("#if"):
            version_match = self.version_pattern.match(stripped)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                version_tuple = (major, minor)
                self.preprocessor_stack.append((True, version_tuple))
                self.version_of_block = version_tuple
            else:
                self.preprocessor_stack.append((False, None))
            return True

        # Track #ifdef and #ifndef directives (not version blocks)
        if stripped.startswith(("#ifdef", "#ifndef")):
            self.preprocessor_stack.append((False, None))
            return True

        # Track #endif directives
        if stripped.startswith("#endif"):
            if self.preprocessor_stack:
                is_version_block, _ = self.preprocessor_stack.pop()
                if is_version_block:
                    self.version_of_block = None
                    for i in range(len(self.preprocessor_stack) - 1, -1, -1):
                        if self.preprocessor_stack[i][0]:
                            self.version_of_block = self.preprocessor_stack[i][1]
                            break
            return True

        # Track #else directives
        # #else replaces the previous #if or #elif, so we pop and push
        if stripped.startswith("#else"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()
            self.preprocessor_stack.append((False, None))
            self.version_of_block = None
            return True

        # Track #elif directives
        if stripped.startswith("#elif"):
            if self.preprocessor_stack:
                self.preprocessor_stack.pop()

            self.version_of_block = None

            version_match_elif = self.version_pattern.match(stripped)
            if version_match_elif:
                major = int(version_match_elif.group(1))
                minor = int(version_match_elif.group(2))
                version_tuple = (major, minor)
                self.preprocessor_stack.append((True, version_tuple))
                self.version_of_block = version_tuple
            else:
                self.preprocessor_stack.append((False, None))
            return True

        return False

    def is_in_version_block(self) -> bool:
        return self.version_of_block is not None

    def get_version_of_block(self) -> tuple[int, int] | None:
        return self.version_of_block


def get_current_version() -> tuple[int, int]:
    """
    Read the current PyTorch (major, minor) from version.txt.

    Uses the same parser as tools/setup_helpers/gen_version_header.py, which
    generates torch/headeronly/version.h from version.h.in.
    """
    version_file = REPO_ROOT / "version.txt"

    if not version_file.exists():
        raise RuntimeError(
            "Could not find version.txt. This linter requires version.txt to run"
        )

    with open(version_file) as f:
        version = f.read().strip()
        major, minor, _patch = parse_version(version)

    return (major, minor)


def get_added_lines(filename: str) -> set[int]:
    """
    Return line numbers (1-indexed) that are new additions in either:
      1. Current uncommitted changes (git diff HEAD), or
      2. Any commit in the current PR (git diff merge-base..HEAD).

    This ensures CI catches issues across all PR commits.
    """
    added_lines: set[int] = set()

    def parse_diff(diff_output: str) -> set[int]:
        lines: set[int] = set()
        current_line = 0
        for line in diff_output.split("\n"):
            # Unified diff format: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith("@@"):
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++"):
                lines.add(current_line)
                current_line += 1
            elif not line.startswith("-"):
                current_line += 1
        return lines

    try:
        # Uncommitted changes
        result = subprocess.run(
            ["git", "diff", "HEAD", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            added_lines.update(parse_diff(result.stdout))

        # Ensure origin/main is up to date before computing merge-base
        result = subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to fetch origin. Error: {result.stderr.strip()}"
            )

        result = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to find merge-base with origin/main. "
                f"Make sure origin/main exists (run 'git fetch origin main'). "
                f"Error: {result.stderr.strip()}"
            )

        merge_base = result.stdout.strip()
        result = subprocess.run(
            ["git", "diff", f"{merge_base}..HEAD", filename],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get git diff information for {filename}. Error: {result.stderr}"
            )
        added_lines.update(parse_diff(result.stdout))

    except Exception as e:
        raise RuntimeError(
            f"Failed to get git diff information for {filename}. Error: {e}"
        ) from e

    return added_lines
