#!/usr/bin/env python3
"""
This lint verifies that test files do not hardcode device strings like "cuda",
"xpu", or "mps" in test functions, to ensure tests can run on any GPU device.

Functions listed in the allowlist file (--allowlist) are exempt.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import re
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "TEST_DEVICE_BIAS"


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


DEVICE_BIAS = ["cuda", "xpu", "mps"]

DEVICE_BIAS_ATTRS = {
    "is_cuda",
    "is_xpu",
    "is_mps",
}


# Matches strings that ARE device identifiers: "cuda", "cuda:0", "xpu:1", etc.
_DEVICE_ID_RE = re.compile(
    r"^(" + "|".join(re.escape(b) for b in DEVICE_BIAS) + r")(:\d+)?$"
)

# For f-string parts: matches a constant part that starts with a device word
# followed by ":" (e.g., the "cuda:" in f"cuda:{idx}")
_FSTRING_BIAS_RE = re.compile(
    r"^(" + "|".join(re.escape(b) for b in DEVICE_BIAS) + r"):"
)


_NOQA_RE = re.compile(r"#\s*noqa\s*:\s*TEST_DEVICE_BIAS\b")


_HELP_URL = "https://pytorch.org/docs/main/notes/device_generic_testing.html"

_SUFFIX = f"If intentional, add '# noqa: TEST_DEVICE_BIAS' to suppress. See {_HELP_URL}"


def _is_device_id(value: str) -> str | None:
    m = _DEVICE_ID_RE.match(value)
    return m.group(1) if m else None


class DeviceBiasVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, allowlist: set[str], lines: list[str]) -> None:
        self.filename = filename
        self.allowlist = allowlist
        self.lint_messages: list[LintMessage] = []
        self._lines = lines
        self._class_stack: list[str] = []
        self._allowlisted_depth: int = 0
        self._allowlisted_key_stack: list[str] = []
        self._used_allowlist_entries: set[str] = set()
        self._allowlist_entry_lines: dict[str, int] = {}
        self._func_depth: int = 0

    def _current_scope(self, func_name: str) -> str:
        if self._class_stack:
            return f"{self._class_stack[-1]}.{func_name}"
        return func_name

    def _is_allowlisted(self, func_name: str) -> bool:
        scope = self._current_scope(func_name)
        key = f"{self.filename}:{scope}"
        return key in self.allowlist

    def _check_node(self, subnode: ast.AST) -> None:
        # String constants that are device identifiers: "cuda", "cuda:0", etc.
        if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
            if _is_device_id(subnode.value):
                self.record(
                    subnode,
                    f"Use torch.accelerator.current_accelerator() or the 'device' "
                    f"parameter from instantiate_device_type_tests instead of "
                    f"'{subnode.value}'. {_SUFFIX}",
                )

        # f-strings: f"cuda:{idx}"
        elif isinstance(subnode, ast.JoinedStr):
            for val in subnode.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    if m := _FSTRING_BIAS_RE.match(val.value):
                        self.record(
                            subnode,
                            f"Use torch.accelerator.current_accelerator() instead of "
                            f"hardcoded '{m.group(1)}' in f-string. {_SUFFIX}",
                        )
                        break

        # Method calls: .cuda(), .xpu(), .mps()
        elif isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Attribute):
            method_name = subnode.func.attr
            if method_name in DEVICE_BIAS:
                self.record(
                    subnode,
                    f"Use .to(torch.accelerator.current_accelerator()) instead of "
                    f".{method_name}(). {_SUFFIX}",
                )

        # Attribute access: .is_cuda, .is_xpu, .is_mps
        elif isinstance(subnode, ast.Attribute):
            if subnode.attr in DEVICE_BIAS_ATTRS:
                self.record(
                    subnode,
                    f"Use device-generic checks instead of .{subnode.attr}. {_SUFFIX}",
                )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._is_allowlisted(node.name):
            key = f"{self.filename}:{self._current_scope(node.name)}"
            self._allowlisted_depth += 1
            self._allowlisted_key_stack.append(key)
            self._allowlist_entry_lines[key] = node.lineno
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1
        if self._is_allowlisted(node.name):
            self._allowlisted_depth -= 1
            self._allowlisted_key_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def generic_visit(self, node: ast.AST) -> None:
        if self._func_depth > 0:
            self._check_node(node)
        super().generic_visit(node)

    def record(self, node: ast.AST, message: str) -> None:
        lineno = getattr(node, "lineno", None)
        if lineno is not None and lineno <= len(self._lines):
            if _NOQA_RE.search(self._lines[lineno - 1]):
                return
        if self._allowlisted_depth > 0:
            self._used_allowlist_entries.add(self._allowlisted_key_stack[-1])
            return
        self.lint_messages.append(
            LintMessage(
                path=self.filename,
                line=lineno,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[device-bias]",
                original=None,
                replacement=None,
                description=message,
            )
        )


def _load_allowlist(allowlist_path: str | None) -> set[str]:
    if not allowlist_path:
        return set()
    entries: set[str] = set()
    with open(allowlist_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.add(line)
    return entries


_global_allowlist: set[str] = set()


def _init_worker(allowlist: set[str]) -> None:
    global _global_allowlist
    _global_allowlist = allowlist


def _normalize_path(filename: str) -> str:
    """Convert absolute paths to repo-relative paths for allowlist matching."""
    import os

    return os.path.relpath(filename)


def check_file(filename: str) -> list[LintMessage]:
    with open(filename) as f:
        source = f.read()
    lines = source.splitlines()
    normalized = _normalize_path(filename)
    tree = ast.parse(source, filename=filename)
    checker = DeviceBiasVisitor(normalized, _global_allowlist, lines)
    checker.visit(tree)

    # Report stale allowlist entries for this file
    file_allowlist = {e for e in _global_allowlist if e.startswith(f"{normalized}:")}
    for entry in sorted(file_allowlist - checker._used_allowlist_entries):
        scope = entry.split(":", 1)[1]
        checker.lint_messages.append(
            LintMessage(
                path=normalized,
                line=checker._allowlist_entry_lines.get(entry),
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[device-bias]",
                original=None,
                replacement=None,
                description=f"'{scope}' no longer has device bias, remove from allowlist",
            )
        )

    return checker.lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect hardcoded device strings in test functions.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    parser.add_argument(
        "--allowlist",
        default=None,
        help="path to allowlist file (one 'filepath:ClassName.method_name' per line)",
    )

    args = parser.parse_args()
    allowlist = _load_allowlist(args.allowlist)

    with mp.Pool(8, initializer=_init_worker, initargs=(allowlist,)) as pool:
        lint_messages = pool.map(check_file, args.filenames)

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
