"""Remote execution support for lint test plans."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from cli.lib.pytorch.lint_test.lint_plans import LintTestPlan
from re_cli.core.core_types import StepConfig
from re_cli.core.job_runner import JobRunner
from re_cli.core.k8s_client import K8sClient, K8sConfig
from re_cli.core.script_builder import RunnerScriptBuilder


logger = logging.getLogger(__name__)

REPO = "https://github.com/pytorch/pytorch.git"
SCRIPT_MODULES_DIR = Path(__file__).resolve().parent / "script_modules"


def _load_script_module(name: str) -> str:
    """Load a bash script from script_modules/ by name (without .sh)."""
    path = SCRIPT_MODULES_DIR / f"{name}.sh"
    if not path.exists():
        raise RuntimeError(f"script module '{name}' not found at {path}")
    template = path.read_text()
    return "\n".join(
        line
        for line in template.splitlines()
        if not line.startswith("#") and not line.startswith("set -")
    ).strip()


def _pr_info(pr: int) -> dict:
    """Get commit SHA and repo URL from a PR number."""
    out = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            "pytorch/pytorch",
            "--json",
            "headRefOid,headRefName,headRepository,headRepositoryOwner",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    owner = data["headRepositoryOwner"]["login"]
    repo_name = data["headRepository"]["name"]
    return {
        "sha": data["headRefOid"],
        "branch": data["headRefName"],
        "repo": f"https://github.com/{owner}/{repo_name}.git",
    }


def _detect_pr() -> int:
    """Detect PR number from the current branch."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            "pytorch/pytorch",
            "--head",
            branch,
            "--json",
            "number",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            "No PR found for current branch. "
            "Push your branch and open a PR first, or pass --pr explicitly."
        )
    prs = json.loads(out.stdout)
    if not prs:
        raise RuntimeError(
            f"No open PR found for branch '{branch}'. "
            "Push your branch and open a PR first, or pass --pr explicitly."
        )
    return prs[0]["number"]


def _local_head() -> str:
    """Get local HEAD commit SHA."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _check_clean_workdir() -> None:
    """Ensure no uncommitted changes exist."""
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    if out.stdout.strip():
        raise RuntimeError(
            "You have uncommitted local changes. "
            "Commit or stash them before running RE."
        )


def _resolve_commit(pr: int | None, commit: str | None) -> dict:
    if commit:
        return {"sha": commit, "repo": REPO}

    _check_clean_workdir()

    if pr is None:
        pr = _detect_pr()
        logger.info("Auto-detected PR #%d from current branch", pr)

    info = _pr_info(pr)
    local_sha = _local_head()
    if local_sha != info["sha"]:
        raise RuntimeError(
            f"Local HEAD ({local_sha[:12]}) does not match "
            f"PR #{pr} HEAD ({info['sha'][:12]}). "
            "Push your changes before running RE."
        )

    logger.info("PR #%d -> %s (%s)", pr, info["sha"][:12], info["repo"])
    return {"sha": info["sha"], "repo": info["repo"]}


def build_run_command(*commands: str) -> str:
    """Wrap commands for execution (venv already activated by setup_uv)."""
    return "\n".join(commands)


class LumenScriptBuilder(RunnerScriptBuilder):
    """RE script builder for lumen lint plans."""

    DEFAULT_MODULES = [
        "header",
        "find_script",
        "git_clone",
        "git_checkout",
        "run_script",
        "upload_outputs",
    ]

    def _add_script(self, module_name: str, script_name: str) -> LumenScriptBuilder:
        body = _load_script_module(script_name)
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: {module_name}\n# {'=' * 44}\n" + body
        )
        return self

    def add_git_clone(self) -> LumenScriptBuilder:
        return self._add_script("git_clone", "git_clone")

    def add_setup_uv(self) -> LumenScriptBuilder:
        return self._add_script("setup_uv", "setup_uv")

    def add_check_python(self) -> LumenScriptBuilder:
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: check_python\n# {'=' * 44}\n"
            "python --version\n"
            "which python"
        )
        return self

    def add_install_lumen(self) -> LumenScriptBuilder:
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: install_lumen\n# {'=' * 44}\n"
            "uv pip install -e .ci/lumen_cli"
        )
        return self


def submit_to_re(
    plan: LintTestPlan,
    commands: list[str],
    pr: int | None,
    commit: str | None,
    dry_run: bool,
    no_follow: bool = False,
    input_overrides: dict[str, str] | None = None,
) -> None:
    """Submit a test plan to Remote Execution."""
    command = build_run_command(*commands)

    # Build runner_modules: fixed prefix + plan bootstrap + fixed suffix.
    # Deduplicate while preserving order.
    modules_list = (
        ["header", "find_script", "git_clone", "git_checkout"]
        + plan.bootstrap
        + ["run_script", "upload_outputs"]
    )
    seen: set[str] = set()
    modules = [m for m in modules_list if not (m in seen or seen.add(m))]

    # Resolve inputs (with overrides) and export as env vars for bootstrap scripts
    resolved_inputs = {**plan.inputs, **(input_overrides or {})}
    env_vars = {
        "PYTHON_VERSION": resolved_inputs.get("python_version", "3.12"),
        **plan.resolve_env_vars(input_overrides),
    }

    # hard code for now for task_type for demoing
    step = StepConfig(
        name=plan.group_id,
        command=command,
        task_type="cpu-large",
        image=plan.image,
        runner_modules=modules,
        env_vars=env_vars,
    )
    job_name = f"{plan.group_id}-pr{pr}" if pr else plan.group_id

    resolved = _resolve_commit(pr, commit)

    # elainewy(should share this with re_cli.core.job_runner.JobRunner for rerun client)
    client = K8sClient(K8sConfig(namespace="remote-execution-system", timeout=60))
    runner = JobRunner(
        client=client,
        name=job_name,
        step_configs=[step],
        script_builder_class=LumenScriptBuilder,
    )
    runner.run(
        commit=resolved["sha"],
        repo=resolved["repo"],
        follow=not no_follow,
        dry_run=dry_run,
    )
    if runner.run_id:
        print(f"\nRun ID: {runner.run_id}")
        print(f"Stream:  blast stream {runner.run_id}")
