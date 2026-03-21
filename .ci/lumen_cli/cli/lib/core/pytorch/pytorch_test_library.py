from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Callable, Union


# ---------------------------------------------------------------------------
# Environment condition types
# ---------------------------------------------------------------------------

# A condition is either:
#   - a plain string tag  → matched as substring of BUILD_ENVIRONMENT
#     e.g. "cuda" matches "pytorch-linux-focal-cuda12.1-py3.10-gcc9-sm86"
#   - a callable          → receives BUILD_ENVIRONMENT string, returns bool
EnvCondition = Union[str, Callable[[str], bool]]

# env_vars can be a static dict or a callable that receives BUILD_ENVIRONMENT
# and returns the dict to apply. Use a callable when different environments
# need different values for the same plan.
#   static:   {"CUDA_VISIBLE_DEVICES": "0"}
#   dynamic:  lambda env: {"HIP_VISIBLE_DEVICES": "0,1,2,3"} if "rocm" in env
#                         else {"CUDA_VISIBLE_DEVICES": "0"}
EnvVarsSpec = Union[dict[str, str], Callable[[str], dict[str, str]]]


def matches_env(condition: EnvCondition, build_env: str) -> bool:
    if callable(condition):
        return condition(build_env)
    return condition in build_env


def resolve_env_vars(spec: EnvVarsSpec, build_env: str) -> dict[str, str]:
    return spec(build_env) if callable(spec) else spec


# ---------------------------------------------------------------------------
# Common pre-built conditions for readability in plan definitions
# ---------------------------------------------------------------------------

def is_cuda(env: str) -> bool:
    return "cuda" in env and "rocm" not in env

def is_rocm(env: str) -> bool:
    return "rocm" in env

def is_xpu(env: str) -> bool:
    return "xpu" in env

def is_gpu(env: str) -> bool:
    return is_cuda(env) or is_rocm(env) or is_xpu(env)

def is_cpu_only(env: str) -> bool:
    return not is_gpu(env)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TestStep:
    test_id: str
    fn: Callable[[], None]
    env_vars: EnvVarsSpec = field(default_factory=dict)
    # Each inner list is a separate pip install invocation, preserving flags and order.
    # e.g. [["--pre", "torchao", "--index-url", "https://..."], ["-e", "."]]
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
    # Step-level setup — runs before this step's fn(), for both normal runs and repro.
    # Executes after plan-level setup_fn, pip_installs, and env_vars are applied.
    setup_fn: Callable[[], None] | None = None


@dataclass
class BasePytorchTestPlan:
    """
    Shared foundation for all PyTorch test plan types.

    run_on:
        Which BUILD_ENVIRONMENT values trigger this plan. Each entry is a
        string tag (substring match) or a callable(build_env) -> bool.
        Empty = eligible on all environments.

    test_configs:
        Which TEST_CONFIG values trigger this plan. Same string/callable
        convention as run_on. Empty = matches any TEST_CONFIG.

    Both conditions must pass for a plan to be selected. This mirrors the
    if/elif dispatch in test.sh where both TEST_CONFIG and BUILD_ENVIRONMENT
    are checked to route to the right test function.

    env_vars:
        Static dict or callable(build_env) -> dict. Step-level wins on conflict.
    """

    group_id: str
    title: str
    steps: list[TestStep]
    env_vars: EnvVarsSpec = field(default_factory=dict)
    pip_installs: list[list[str]] = field(default_factory=list)
    working_dir: str | None = None
    # Always runs before any step, whether executing the full plan or reproducing
    # a single step. If setup is required for a test to run, it belongs here.
    setup_fn: Callable[[], None] | None = None
    # BUILD_ENVIRONMENT filter — empty = all environments
    run_on: list[EnvCondition] = field(default_factory=list)
    # TEST_CONFIG filter — empty = all configs
    test_configs: list[EnvCondition] = field(default_factory=list)

    def is_eligible(self, build_env: str, test_config: str = "") -> bool:
        env_ok = not self.run_on or any(
            matches_env(c, build_env) for c in self.run_on
        )
        config_ok = not self.test_configs or any(
            matches_env(c, test_config) for c in self.test_configs
        )
        return env_ok and config_ok


# ---------------------------------------------------------------------------
# Concrete plan types
# ---------------------------------------------------------------------------

@dataclass
class CoreTestPlan(BasePytorchTestPlan):
    """Standard python test/run_test.py based tests."""
    pass


@dataclass
class BenchmarkTestPlan(BasePytorchTestPlan):
    """
    Inductor / dynamo benchmark tests, one TestStep per model.

    __post_init__ auto-generates one step per model plus a final _join_results
    step. This lets a single failing model be reproduced in isolation:

        python -m cli.run test pytorch-core \\
            --group-id pytorch_inductor_smoketest --test-id BERT_pytorch
    """

    device: str = "cuda"
    backend: str = "inductor"
    modes: list[str] = field(default_factory=list)  # e.g. ["training", "inference"]
    dtype: str = "amp"
    suite: str = "torchbench"  # torchbench | huggingface | timm_models
    models: list[str] = field(default_factory=list)
    output_dir: str = "test/test-reports"
    extra_benchmark_flags: list[str] = field(default_factory=list)
    # Called once after all per-model CSVs exist; receives per-model output paths.
    join_results_fn: Callable[[list[str]], None] | None = None

    def __post_init__(self) -> None:
        if self.models:
            self.steps = self._build_steps()

    def _per_model_output_path(self, model: str) -> str:
        mode_tag = "_".join(self.modes) if self.modes else "default"
        return os.path.join(
            self.output_dir,
            f"{self.backend}_{self.suite}_{model}_{mode_tag}_{self.device}.csv",
        )

    def _run_model(self, model: str) -> None:
        from cli.lib.core.pytorch.run_test_helper import run_command_checked

        flags = [
            f"--device {self.device}",
            f"--backend {self.backend}",
            f"--only {model}",
            f"--output {self._per_model_output_path(model)}",
        ] + self.extra_benchmark_flags

        for mode in self.modes or []:
            run_command_checked(
                f"python benchmarks/dynamo/{self.suite}.py "
                + " ".join(flags)
                + f" --{mode} --{self.dtype}"
            )

    def _join_results(self) -> None:
        if self.join_results_fn is None:
            return
        self.join_results_fn([self._per_model_output_path(m) for m in self.models])

    def _build_steps(self) -> list[TestStep]:
        steps = [
            TestStep(test_id=model, fn=functools.partial(self._run_model, model))
            for model in self.models
        ]
        steps.append(TestStep(test_id="_join_results", fn=self._join_results))
        return steps
