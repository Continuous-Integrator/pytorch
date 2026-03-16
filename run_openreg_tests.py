#!/usr/bin/env python3
"""Run PyTorch's device-generic test suite against the openreg backend.

This script serves as both the CI launcher for openreg and a template for
out-of-tree PrivateUse1 backends to run PyTorch's device-generic tests.

Usage:
    python run_openreg_tests.py                          # run full allowlist
    python run_openreg_tests.py test_torch.py            # run specific file(s)
    python run_openreg_tests.py --list                   # print allowlist
    python run_openreg_tests.py -c                       # don't stop on first failure
    python run_openreg_tests.py --timeout 30             # per-test timeout in seconds
    python run_openreg_tests.py --retries 3              # retry failed tests N times

Prerequisites:
    - PyTorch must be built and installed
    - torch_openreg will be auto-installed if missing
"""

import argparse
from collections import defaultdict
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import sysconfig
import threading
import time

PYTORCH_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PYTORCH_ROOT, "test")
OPENREG_DIR = os.path.join(
    PYTORCH_ROOT,
    "test",
    "cpp_extensions",
    "open_registration_extension",
    "torch_openreg",
)

# Device-generic tests that openreg supports.
ALLOWLIST = [
    "nn/test_convolution.py",
    "nn/test_dropout.py",
    "nn/test_embedding.py",
    "nn/test_init.py",
    "nn/test_multihead_attention.py",
    "nn/test_parametrization.py",
    "nn/test_pooling.py",
    "test_autograd.py",
    "test_binary_ufuncs.py",
    "test_custom_ops.py",
    "test_dataloader.py",
    "test_indexing.py",
    "test_masked.py",
    "test_modules.py",
    "test_native_mha.py",
    "test_nn.py",
    "test_ops.py",
    "test_ops_fwd_gradients.py",
    "test_ops_gradients.py",
    "test_optim.py",
    "test_reductions.py",
    "test_scatter_gather_ops.py",
    "test_segment_reductions.py",
    "test_serialization.py",
    "test_shape_ops.py",
    "test_sort_and_select.py",
    "test_tensor_creation_ops.py",
    "test_testing.py",
    "test_torch.py",
    "test_transformers.py",
    "test_type_promotion.py",
    "test_unary_ufuncs.py",
    "test_utils.py",
    "test_view_ops.py",
]


def install_openreg() -> str:
    """Install torch_openreg and return the install site-packages path.

    Follows the same approach as test/run_test.py's install_cpp_extensions:
    pip install into a local --root so we can add it to PYTHONPATH for
    subprocesses without polluting the global environment.
    """
    build_dir = os.path.join(OPENREG_DIR, "build")
    install_root = os.path.join(OPENREG_DIR, "install")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    if os.path.exists(install_root):
        shutil.rmtree(install_root)

    print(f"Installing torch_openreg from {OPENREG_DIR} ...")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation", ".", "--root", "./install",
        ],
        cwd=OPENREG_DIR,
    )

    platlib = sysconfig.get_paths()["platlib"]
    platlib_rel = os.path.relpath(platlib, os.path.splitdrive(platlib)[0] + os.sep)
    install_dir = os.path.join(install_root, platlib_rel)

    # Smoke test in a subprocess (avoids circular import).
    subprocess.check_call(
        [
            sys.executable, "-c",
            "import torch, torch_openreg; "
            "print(f'PyTorch: {torch.__version__}'); "
            "print(f'openreg device count: {torch.openreg.device_count()}'); "
            "print(f'Backend registered: {torch._C._get_privateuse1_backend_name()}')",
        ],
        env={**os.environ, "PYTHONPATH": install_dir},
    )
    return install_dir


def _log_dir() -> str:
    log_dir = os.path.join(PYTORCH_ROOT, "openreg_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _log_path(test_file: str) -> str:
    # nn/test_dropout.py -> nn__test_dropout.log
    return os.path.join(_log_dir(), test_file.replace("/", "__").replace(".py", ".log"))


def _parse_skipped_tests(log_file: str) -> list[dict[str, str]]:
    """Parse pytest -rs output for skipped tests.

    Returns a list of {"reason": ..., "location": ...} dicts from the
    short test summary section. These tell you which tests were skipped
    and why (e.g. "Only runs on ['cuda']").
    """
    skipped: list[dict[str, str]] = []
    try:
        with open(log_file, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
    except OSError:
        return skipped

    # Pytest -rs short summary lines look like:
    #   SKIPPED [1] torch/testing/_internal/common_device_type.py:367: Only runs on ['cuda']
    for match in re.finditer(
        r"^SKIPPED \[\d+\] (.+?): (.+)$", content, re.MULTILINE
    ):
        skipped.append({
            "location": match.group(1).strip(),
            "reason": match.group(2).strip(),
        })

    return skipped


def _read_stepcurrent(stepcurrent_key: str) -> str | None:
    """Read the last-run test nodeid from the pytest stepcurrent cache."""
    cache_file = os.path.join(
        PYTORCH_ROOT, ".pytest_cache/v/cache/stepcurrent", stepcurrent_key, "lastrun"
    )
    try:
        with open(cache_file) as f:
            return f.read()
    except FileNotFoundError:
        return None


def _run_and_tee(command, cwd, env, log_file, timeout):
    """Run command, tee stdout+stderr to both the terminal and a log file.

    Returns the exit code. Raises subprocess.TimeoutExpired on timeout.
    On timeout: sends SIGINT, waits 5s for graceful shutdown, then kills.
    Follows the pattern from torch/testing/_internal/common_utils.py wait_for_process.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def reader():
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        t.join(timeout=5)
        raise

    t.join(timeout=10)
    return proc.returncode


def run_test(
    test_file: str,
    openreg_pythonpath: str,
    timeout: int,
    retries: int,
) -> tuple[str, float, list[str], list[str], list[dict[str, str]]]:
    """Run a single test file with timeout and per-test retry.

    Returns (status, elapsed, consistent_failures, flaky_failures, skipped).

    Uses the stepcurrent mechanism (same as CI's run_test.py) to retry
    individual failing test methods rather than the entire file.

    status is one of: "PASS", "FAIL", "FLAKY", "TIMEOUT"
    """
    full_path = os.path.join(TEST_DIR, test_file)
    pythonpath = openreg_pythonpath
    if "PYTHONPATH" in os.environ:
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": "openreg",
        "OPENREG_DISABLE_FALLBACK_BLOCKLIST": "1",
        "OPENREG_DISABLE_MEMORY_PROTECTION": "1",
    }

    log_file = _log_path(test_file)
    print(f"\n{'=' * 60}")
    print(f"Running {test_file}  (log: {log_file})")
    print("=" * 60, flush=True)

    command = [sys.executable, "-u", full_path, "--use-pytest", "-v", "-x", "-rs"]
    stepcurrent_key = f"openreg_{test_file.replace('/', '_').replace('.py', '')}"
    sc_command = f"--sc={stepcurrent_key}"
    num_failures: dict[str, int] = defaultdict(int)

    start = time.monotonic()

    with open(log_file, "wb") as lf:
        while True:
            try:
                ret_code = _run_and_tee(
                    command + [sc_command],
                    cwd=TEST_DIR,
                    env=env,
                    log_file=lf,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                print(f"  TIMEOUT  {test_file}  ({elapsed:.1f}s, exceeded {timeout}s limit)")
                return "TIMEOUT", elapsed, [], [], []

            # Exit code 5 means "no tests collected", treat as pass
            ret_code = 0 if ret_code == 5 else ret_code

            if ret_code == 0 and not sc_command.startswith("--rs="):
                # Reached end of test suite successfully
                break

            current_failure = _read_stepcurrent(stepcurrent_key)
            if current_failure is None:
                if ret_code != 0:
                    print("  No stepcurrent file found (possible import error)")
                break

            if ret_code != 0:
                num_failures[current_failure] += 1

            if ret_code == 0:
                # Rerunning the previously failing test passed — it was flaky
                sc_command = f"--scs={stepcurrent_key}"
                print("  Test succeeded on rerun, continuing with remaining tests")
            elif num_failures[current_failure] >= retries:
                print(f"  FAILED CONSISTENTLY: {current_failure}")
                # Skip the consistently failing test and continue
                sc_command = f"--scs={stepcurrent_key}"
            else:
                # Retry the single failing test
                sc_command = f"--rs={stepcurrent_key}"
                print(
                    f"  Retrying {current_failure} "
                    f"({num_failures[current_failure]}/{retries}) ..."
                )

    elapsed = time.monotonic() - start

    # Strip quotes added by stepcurrent cache format
    strip = lambda s: s.strip('"')
    consistent_failures = [strip(t) for t, n in num_failures.items() if n >= retries]
    flaky_failures = [strip(t) for t, n in num_failures.items() if 0 < n < retries]
    skipped = _parse_skipped_tests(log_file)

    if consistent_failures:
        print(f"  FAIL  {test_file}  ({elapsed:.1f}s)")
        print(f"    Consistent failures: {consistent_failures}")
        if flaky_failures:
            print(f"    Flaky (passed on retry): {flaky_failures}")
        return "FAIL", elapsed, consistent_failures, flaky_failures, skipped
    elif flaky_failures:
        print(f"  FLAKY  {test_file}  ({elapsed:.1f}s)")
        print(f"    Flaky (passed on retry): {flaky_failures}")
        return "FLAKY", elapsed, [], flaky_failures, skipped
    else:
        print(f"  PASS  {test_file}  ({elapsed:.1f}s)")
        return "PASS", elapsed, [], [], skipped


def print_summary(results: list[tuple[str, str, float]]) -> None:
    """Print a summary table of test results."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    name_width = max(len(name) for name, _, _ in results)
    for name, status, elapsed in results:
        log_file = _log_path(name)
        print(f"  {status:<7}  {name:<{name_width}}  ({elapsed:.1f}s)  {log_file}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    flaky = sum(1 for _, s, _ in results if s == "FLAKY")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    timed_out = sum(1 for _, s, _ in results if s == "TIMEOUT")
    total_time = sum(t for _, _, t in results)

    parts = [f"{passed} passed"]
    if flaky:
        parts.append(f"{flaky} flaky")
    if failed:
        parts.append(f"{failed} failed")
    if timed_out:
        parts.append(f"{timed_out} timed out")
    print(f"\n{', '.join(parts)}  ({total_time:.1f}s total)")
    print(f"Logs: {_log_dir()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyTorch device-generic tests against the openreg backend."
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific test files to run (default: full allowlist).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the allowlist and exit.",
    )
    parser.add_argument(
        "-c",
        "--continue-on-failure",
        action="store_true",
        help="Continue running tests after a failure (default: stop on first failure).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-test timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times a test must fail before it's considered a consistent failure (default: 3).",
    )
    args = parser.parse_args()

    if args.list:
        for f in ALLOWLIST:
            print(f)
        return

    test_files = args.tests if args.tests else ALLOWLIST

    openreg_pythonpath = install_openreg()

    results: list[tuple[str, str, float]] = []
    all_consistent_failures: list[str] = []
    all_flaky_failures: list[str] = []
    all_skipped: list[dict[str, str]] = []
    for test_file in test_files:
        status, elapsed, consistent, flaky, skipped = run_test(
            test_file, openreg_pythonpath, args.timeout, args.retries
        )
        results.append((test_file, status, elapsed))
        all_consistent_failures.extend(consistent)
        all_flaky_failures.extend(flaky)
        all_skipped.extend(skipped)
        if status in ("FAIL", "TIMEOUT") and not args.continue_on_failure:
            print(f"\nStopping after failure in {test_file}.")
            break

    print_summary(results)

    # Deduplicate skipped entries and group by reason for readability
    skipped_by_reason: dict[str, int] = defaultdict(int)
    for entry in all_skipped:
        skipped_by_reason[entry["reason"]] += 1

    report = {
        "failed": all_consistent_failures,
        "flaky": all_flaky_failures,
        "timed_out": [name for name, s, _ in results if s == "TIMEOUT"],
        "skipped_by_reason": dict(skipped_by_reason),
    }
    # Write to test/test-reports/ so CI's upload-test-artifacts picks it up
    report_dir = os.path.join(PYTORCH_ROOT, "test", "test-reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "openreg_device_generic_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")

    if any(s in ("FAIL", "TIMEOUT") for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
