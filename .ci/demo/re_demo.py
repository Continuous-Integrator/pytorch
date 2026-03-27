#!/usr/bin/env python3
"""
Rerun PyTorch CI jobs via blast Remote Execution.

Usage:
  # Dry run to see the generated script:
  python re_demo.py run .github/workflows/lint-osdc.yml -j lintrunner-noclang --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


try:
    import yaml
    import re_cli  # noqa: F401
except ImportError:
    print("Missing dependencies. Install with:blast-cli pyyaml")
    sys.exit(1)

from re_cli.core.core_types import StepConfig
from re_cli.core.job_runner import JobRunner
from re_cli.core.k8s_client import K8sClient, K8sConfig
from re_cli.core.script_builder import RunnerScriptBuilder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = "https://github.com/pytorch/pytorch.git"
IMAGE = "ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930"
GHA_EXPR = re.compile(r"\$\{\{.*?\}\}")
# action_key → this mimic some customize action script which is critical step to run
ACTION_SCRIPTS = {
    "pytorch/test-infra/.github/actions/setup-uv": "setup_uv.sh",
}
ACTION_MAP = {
    key: lambda inputs, s=script: inline_action(s, inputs)
    for key, script in ACTION_SCRIPTS.items()
}
ACTIONS_DIR = Path(__file__).resolve().parent / "actions"


# ---------------------------------------------------------------------------
# Script builder：customize the script builder to add customized steps
# ---------------------------------------------------------------------------
class PyTorchScriptBuilder(RunnerScriptBuilder):
    DEFAULT_MODULES = [
        "header",
        "find_script",
        "git_clone",
        "git_checkout",
        "run_script",
        "upload_outputs",
    ]

    # just example to override the script builder from origincal cli tool
    # the RunnerScriptBuilder automatically catch the module name with add_xxx() method
    def add_git_clone(self) -> "PyTorchScriptBuilder":
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: git_clone\n# {'=' * 44}\n"
            'if [[ -n "$GIT_REPO" ]]; then\n'
            '    echo "[Runner] Cloning $GIT_REPO..."\n'
            '    git clone --depth=1 "$GIT_REPO" repo\n'
            "    cd repo\n"
            '    REPO_DIR="$(pwd)"\n'
            "    export REPO_DIR\n"
            '    echo "[Runner] REPO_DIR=$REPO_DIR"\n'
            "else\n"
            '    echo "[Runner] No git repo specified, skipping clone"\n'
            "fi\n"
        )
        return self

# ---------------------------------------------------------------------------
# Action → bash mapping
# ---------------------------------------------------------------------------
def inline_action(script_name: str, inputs: dict) -> str:
    """Inline a bash script from actions/ with all with: inputs auto-exported."""
    template = (ACTIONS_DIR / script_name).read_text()
    body = "\n".join(
        line
        for line in template.splitlines()
        if not line.startswith("#") and not line.startswith("set -")
    ).strip()
    exports = "\n".join(
        f'export {k.replace("-", "_").upper()}="{str(v).lower() if isinstance(v, bool) else v}"'
        for k, v in inputs.items()
    )
    return f"{exports}\n{body}" if exports else body

# pass the workflow yaml to get the job info
# for this demo, we only care about the job with `uses:` and `with.script`
def extract_setup_steps(uses: str) -> list[dict]:
    """Extract setup steps from a reusable workflow.
    - run: steps with `# re:add` → include bash directly
    - uses: steps in ACTION_MAP → convert to bash
    """
    if not uses.startswith("./"):
        return []
    resolved = Path.cwd() / uses.split("@")[0].removeprefix("./")
    if not resolved.exists():
        return []
    wf = yaml.safe_load(resolved.read_text())
    substeps = []
    for _job_name, job_def in wf.get("jobs", {}).items():
        for step in job_def.get("steps", []):
            name = step.get("name", f"step-{len(substeps)}")

            run = step.get("run")
            if run and "# re:add" in run:
                substeps.append({"name": name, "bash": run.strip()})
                continue

            action_key = step.get("uses", "").split("@")[0]
            if action_key in ACTION_MAP:
                bash = ACTION_MAP[action_key](step.get("with", {}))
                substeps.append({"name": name, "bash": bash})
            elif action_key:
                print(
                    f"  Warning: skipping unmapped action '{action_key}' (step: {name})"
                )
    return substeps

def parse_workflow_jobs(path: str) -> dict[str, dict]:
    """Parse jobs with `uses:` + `with.script` from a workflow file."""
    wf = yaml.safe_load(Path(path).read_text())
    jobs = {}
    for name, defn in wf.get("jobs", {}).items():
        uses = defn.get("uses", "")
        if not uses:
            continue
        w = defn.get("with", {})
        script = w.get("script", "")
        if not script:
            continue
        jobs[name] = {
            "image": w.get("docker-image", ""),
            "script": script,
            "uses": uses,
            "has_gha_expr": bool(GHA_EXPR.search(script)),
        }
    return jobs


def build_command(setup_steps: list[dict], cmd: str) -> str:
    parts = ["#!/bin/bash", "set -e", ""]
    for s in setup_steps:
        parts.append(f"# === {s['name']} ===")
        parts.append(s["bash"].strip())
        parts.append("")
    parts.append("# === run ===")
    parts.append(cmd.strip())
    parts.append("")
    return "\n".join(parts)


# extract commit and repo info from PR
def pr_info(pr: int) -> dict:
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


def resolve(args) -> dict:
    if args.pr:
        info = pr_info(args.pr)
        print(f"PR #{args.pr} -> {info['sha'][:12]} ({info['repo']})")
        return {"sha": info["sha"], "repo": info["repo"]}
    if args.commit:
        return {"sha": args.commit, "repo": REPO}
    print("Provide --pr or --commit")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
def submit(steps: list[StepConfig], name: str, args):
    client = K8sClient(K8sConfig(namespace="remote-execution-system", timeout=60))
    resolved = resolve(args)
    runner = JobRunner(
        client=client,
        name=name,
        step_configs=steps,
        script_builder_class=PyTorchScriptBuilder,
    )
    runner.run(
        commit=resolved["sha"],
        repo=resolved["repo"],
        follow=not args.no_follow,
        dry_run=args.dry_run,
    )
    if runner.run_id:
        print(f"\nRun ID: {runner.run_id}")
        print(f"Stream:  blast stream {runner.run_id}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(args):
    setup_steps = []
    image = args.image or IMAGE
    cmd = args.cmd

    if args.workflow and args.job:
        jobs = parse_workflow_jobs(args.workflow)
        if args.job not in jobs:
            print(f"Job '{args.job}' not found. Available: {', '.join(jobs)}")
            sys.exit(1)

        job = jobs[args.job]
        image = args.image or job["image"] or IMAGE
        setup_steps = extract_setup_steps(job["uses"])

        if not cmd:
            if job["has_gha_expr"]:
                print("Script contains GHA expressions that can't run outside CI:\n")
                print("  " + job["script"].replace("\n", "\n  "))
                print("\nProvide --cmd with your adapted command. Example:")
                clean = GHA_EXPR.sub('"*"', job["script"])
                print(f"  --cmd '{clean.strip().splitlines()[-1].strip()}'")
                sys.exit(1)
            cmd = job["script"]
    elif args.workflow:
        print("--workflow requires --job / -j")
        sys.exit(1)

    if not cmd:
        print("Provide --cmd or --workflow + --job")
        sys.exit(1)

    command = build_command(setup_steps, cmd)
    step = StepConfig(
        name=args.job or "run",
        command=command,
        task_type="cpu-large",
        image=image,
    )
    job_name = f"{step.name}-pr{args.pr}" if args.pr else step.name
    submit([step], job_name, args)


def cmd_list(args):
    jobs = parse_workflow_jobs(args.workflow)
    if not jobs:
        print(f"No runnable jobs in {args.workflow}")
        sys.exit(1)

    print(f"Jobs in {args.workflow}:\n")
    for name, info in jobs.items():
        tag = info["image"].split(":")[-1][:30] if info["image"] else "-"
        dynamic = " (has ${{}})" if info["has_gha_expr"] else ""
        print(f"  {name:<35} {tag}{dynamic}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Run a CI job on RE")
    r.add_argument("workflow", nargs="?", help="Path to workflow YAML")
    r.add_argument("-j", "--job", help="Job name from the workflow")
    r.add_argument("--cmd", help="Command to run (overrides workflow script)")
    r.add_argument("--image", help="Override Docker image")
    r.add_argument("--pr", type=int, help="PR number")
    r.add_argument("--commit", help="Commit SHA")
    r.add_argument("--patch", action="store_true", help="Include local changes")
    r.add_argument("--dry-run", action="store_true", help="Show what would run")
    r.add_argument("--no-follow", action="store_true", help="Don't stream logs")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="List jobs in a workflow")
    ls.add_argument("workflow", help="Path to workflow YAML")
    ls.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
