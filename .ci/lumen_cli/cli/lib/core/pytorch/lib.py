from __future__ import annotations

import logging
import os

from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import run_command, temp_environ, working_directory
from cli.lib.core.pytorch.pytorch_test_library import (
    BasePytorchTestPlan,
    TestStep,
    resolve_env_vars,
)
from cli.lib.core.pytorch.plans.benchmark_tests import BENCHMARK_TEST_PLANS
from cli.lib.core.pytorch.plans.core_tests import CORE_TEST_PLANS


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Central registry — add new plan files here as they are created
# ---------------------------------------------------------------------------

PYTORCH_TEST_LIBRARY: dict[str, BasePytorchTestPlan] = {
    **CORE_TEST_PLANS,
    **BENCHMARK_TEST_PLANS,
}


def resolve_plans_for_env(
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> list[str]:
    """Return group_ids eligible for the given build_env (ignores TEST_CONFIG)."""
    registry = library if library is not None else PYTORCH_TEST_LIBRARY
    return [gid for gid, plan in registry.items() if plan.is_eligible(build_env)]


def resolve_plan_for_test_config(
    test_config: str,
    build_env: str,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> str:
    """
    Resolve exactly one group_id from TEST_CONFIG + build_env.

    Replaces the if/elif dispatch at the bottom of test.sh. In test.sh:

        (cd .ci/lumen_cli && python -m pip install -e .)
        python -m cli.run test pytorch-core \\
            --test-config "$TEST_CONFIG" \\
            --build-env  "$BUILD_ENVIRONMENT" \\
            --shard-id   "$SHARD_NUMBER" \\
            --num-shards "$NUM_TEST_SHARDS"

    Raises RuntimeError if zero or more than one plan matches — ambiguity is
    a configuration error that should be fixed in the plan definitions.
    """
    registry = library if library is not None else PYTORCH_TEST_LIBRARY
    matched = [
        gid for gid, plan in registry.items()
        if plan.is_eligible(build_env, test_config)
    ]

    if not matched:
        raise RuntimeError(
            f"No plan matched TEST_CONFIG={test_config!r} build_env={build_env!r}. "
            f"Available group_ids: {sorted(registry)}"
        )
    if len(matched) > 1:
        raise RuntimeError(
            f"Ambiguous match for TEST_CONFIG={test_config!r} build_env={build_env!r}: "
            f"{matched}. Tighten test_configs/run_on conditions."
        )
    return matched[0]


# ---------------------------------------------------------------------------
# Repro context + runner
# ---------------------------------------------------------------------------

def _build_repro_context(
    plan: BasePytorchTestPlan,
    step: TestStep,
    build_env: str,
) -> dict:
    """
    Build the reproducible context for a step.

    env_vars are kept separate (not merged) so the runner can apply them at
    the right scope:
      - plan_env_vars: set once around all steps, persist for the whole plan
      - step_env_vars: scoped to this step only, cleaned up after it finishes
    """
    return {
        "plan_env_vars": resolve_env_vars(plan.env_vars, build_env),
        "step_env_vars": resolve_env_vars(step.env_vars, build_env),
        "pip_installs": plan.pip_installs + step.pip_installs,
        "working_dir": step.working_dir or plan.working_dir,
    }


def _print_repro(group_id: str, step: TestStep, ctx: dict) -> None:
    lines = [f"\nTo reproduce {group_id}/{step.test_id}:"]
    for pip_args in ctx["pip_installs"]:
        lines.append(f"  pip install {' '.join(pip_args)}")
    if ctx["working_dir"]:
        lines.append(f"  cd {ctx['working_dir']}")
    for k, v in ctx["plan_env_vars"].items():
        lines.append(f"  export {k}={v}  # plan-level")
    for k, v in ctx["step_env_vars"].items():
        lines.append(f"  export {k}={v}  # step-level (scoped)")
    lines.append(
        f"  python -m cli.run test pytorch-core"
        f" --group-id {group_id} --test-id {step.test_id}"
    )
    logger.error("\n".join(lines))


def run_test_plan(
    group_id: str,
    build_env: str,
    test_id: str | None = None,
    cmd: str | None = None,
    shard_id: int = 1,
    num_shards: int = 1,
    library: dict[str, BasePytorchTestPlan] | None = None,
) -> None:
    """
    Run a test plan (or a single step within it) from the registry.

    Args:
        group_id:  Key in PYTORCH_TEST_LIBRARY, e.g. "pytorch_cpuonly".
        test_id:   If set, run only that step — used for single-step reproduction.
        cmd:       If set, replay the setup context of test_id but run this
                   command instead of step.fn(). Requires test_id to be set.
                   Use this to reproduce a specific pytest line within a step.
        shard_id:  Current shard (1-based).
        num_shards: Total number of shards.
        library:   Override the default registry (useful for testing).
    """
    if cmd and not test_id:
        raise RuntimeError("--cmd requires --test-id to identify the setup context")
    registry = library if library is not None else PYTORCH_TEST_LIBRARY

    if group_id not in registry:
        raise RuntimeError(
            f"group '{group_id}' not found. Available: {sorted(registry)}"
        )
    plan = registry[group_id]

    steps = (
        [s for s in plan.steps if s.test_id == test_id]
        if test_id
        else plan.steps
    )
    if not steps:
        raise RuntimeError(
            f"test_id '{test_id}' not found in '{group_id}'. "
            f"Available: {[s.test_id for s in plan.steps]}"
        )

    if plan.setup_fn:
        logger.info("[%s] running setup", group_id)
        plan.setup_fn()

    failures: list[str] = []

    # Plan-level env_vars wrap all steps — set once, persist across the whole plan.
    first_ctx = _build_repro_context(plan, steps[0], build_env)
    with temp_environ(first_ctx["plan_env_vars"]):
        for step in steps:
            ctx = _build_repro_context(plan, step, build_env)
            logger.info("[%s/%s] starting", group_id, step.test_id)

            for pip_args in ctx["pip_installs"]:
                pip_install_packages(pip_args)

            # Step-level env_vars are scoped to this step only.
            with (
                temp_environ(ctx["step_env_vars"]),
                working_directory(ctx["working_dir"] or ""),
            ):
                try:
                    if step.setup_fn:
                        logger.info("[%s/%s] running step setup", group_id, step.test_id)
                        step.setup_fn()
                    if cmd:
                        logger.info("[%s/%s] running custom cmd: %s", group_id, step.test_id, cmd)
                        run_command(cmd, use_shell=True)
                    else:
                        step.fn()
                    logger.info("[%s/%s] passed", group_id, step.test_id)
                except Exception as e:
                    logger.error("[%s/%s] FAILED: %s", group_id, step.test_id, e)
                    _print_repro(group_id, step, ctx)
                    failures.append(step.test_id)

    if failures:
        raise RuntimeError(
            f"[{group_id}] {len(failures)} step(s) failed: {failures}"
        )
