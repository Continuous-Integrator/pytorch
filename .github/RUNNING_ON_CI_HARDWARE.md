# Using PyTorch CI as Remote Hardware

Sometimes you need to run something on hardware you don't have:

- Reproducing a test that fails on a specific runner you don't own
  (different GPU SKU, CUDA version, driver, OS, compiler)
- Validating a new MPS kernel without a Mac
- Timing a CUDA kernel on a specific GPU SKU
- Adding `print` statements or running under `compute-sanitizer` to see
  what happens on CI hardware
- Bisecting a regression that only manifests on CI

## First: can you reuse an existing workflow?

Before writing a new one, check whether an existing workflow under
`.github/workflows/` already targets the hardware/build-env you want.
Many workflows opt-in to PR runs via a `ciflow/<label>/*` tag trigger,
e.g. `h100-distributed.yml`:

```yaml
on:
  push:
    tags:
      - ciflow/h100-distributed/*
```

To trigger that workflow on your PR, push a tag matching the pattern:

```bash
git tag ciflow/h100-distributed/<your-pr-number>
git push origin ciflow/h100-distributed/<your-pr-number>
```

(ghstack does this automatically for any `ciflow/*` tags found in the
commit when you `ghstack`.) Existing tag families include `ciflow/h100`,
`ciflow/h100-distributed`, `ciflow/inductor`, `ciflow/rocm`,
`ciflow/slow`, `ciflow/trunk`, and more — `grep -r "ciflow/" .github/workflows/`
to find the full list.

You can also retrigger a previously-run workflow without a push via
`workflow_dispatch`:

```bash
gh workflow run <name>.yml --ref <your-branch>
```

If a `ciflow` tag or `workflow_dispatch` covers your case, stop here.
The recipe below is for cases where you need a *new* config (a runner +
test command combination that no existing workflow exposes), or where
you need to suppress unrelated workflows on the PR.

## Recipe: minimal new workflow

A worked example is PR
[#181754](https://github.com/pytorch/pytorch/pull/181754) (reproducing
flaky test [#181685](https://github.com/pytorch/pytorch/issues/181685)).

### 1. Identify the runner and build env you want

If you're chasing a CI failure, pull the failing job's log to capture
exactly what to mirror.

Auto-disable issues link to a workflow page like
`https://github.com/pytorch/pytorch/runs/<id>`. That URL is a
**check-run**, not a workflow run — `gh run view` will 404 on it.

```bash
# Resolve check-run -> actions job ID. In PyTorch's setup the check-run
# ID and the actions job ID are the same number; the html_url contains
# both: https://github.com/.../actions/runs/<run_id>/job/<job_id>
gh api repos/pytorch/pytorch/check-runs/<id> | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("name:    ", d["name"])
print("html_url:", d["html_url"])
print("head_sha:", d["head_sha"])
'

# Download the raw job log
gh api repos/pytorch/pytorch/actions/jobs/<id>/logs > /tmp/job.log
```

From the log (or from existing workflow YAML), capture:

- **Runner type** (e.g. `linux.g5.4xlarge.nvidia.gpu`,
  `macos-m1-stable`, `linux.rocm.gpu.gfx942.4`)
- **Build environment** (e.g. `linux-jammy-cuda13.0-py3.10-gcc11-sm86`)
- **Docker image** (e.g.
  `ci-image:pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11-inductor-benchmarks`)
- **Exact command** to run, if reproducing

Existing workflows under `.github/workflows/` are the easiest source
for build-env / docker-image / runner combinations. Find one that uses
your target hardware and copy its build config.

### 2. Add a minimal workflow

Create `.github/workflows/<name>.yml`. Mirror the build env from step 1
and a single test config. The `concurrency:` block is important — without
it, every push spawns a new full build instead of canceling the prior
run, exactly what you don't want for a debug-loop PR.

```yaml
name: <name>

on:
  push:
    branches:
      - '<name>*'
  pull_request:
    paths:
      - .github/workflows/<name>.yml
      - .ci/pytorch/test.sh
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
  cancel-in-progress: true

permissions:
  id-token: write
  contents: read
  actions: read

jobs:
  get-label-type:
    name: get-label-type
    uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main
    if: github.repository_owner == 'pytorch'
    with:
      triggering_actor: ${{ github.triggering_actor }}
      issue_owner: ${{ github.event.pull_request.user.login || github.event.issue.user.login }}
      curr_branch: ${{ github.head_ref || github.ref_name }}
      curr_ref_type: ${{ github.ref_type }}
      # Don't route through the experimental Linux Foundation runner pool;
      # use the AWS pool the rest of the repo uses.
      opt_out_experiments: lf

  build:
    uses: ./.github/workflows/_linux-build.yml
    needs: get-label-type
    with:
      build-environment: linux-jammy-cuda13.0-py3.10-gcc11-sm86
      docker-image-name: ci-image:pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11-inductor-benchmarks
      cuda-arch-list: '8.6'
      runner_prefix: "${{ needs.get-label-type.outputs.label-type }}"
      test-matrix: |
        { include: [
          { config: "<name>", shard: 1, num_shards: 1, runner: "${{ needs.get-label-type.outputs.label-type }}linux.g5.4xlarge.nvidia.gpu" },
        ]}
    secrets: inherit

  test:
    uses: ./.github/workflows/_linux-test.yml
    needs: build
    with:
      build-environment: ${{ needs.build.outputs.build-environment }}
      docker-image: ${{ needs.build.outputs.docker-image }}
      test-matrix: ${{ needs.build.outputs.test-matrix }}
    secrets: inherit
```

For non-Linux: `_mac-build.yml` / `_mac-test.yml`, `_win-build.yml` /
`_win-test.yml`, `_rocm-test.yml`.

### 3. Add a `TEST_CONFIG` branch in `.ci/pytorch/test.sh`

The `config: "<name>"` from step 2 becomes the `TEST_CONFIG` env var on
the runner. Find the dispatch chain (`elif [[ "${TEST_CONFIG}" == ...`)
and add a new branch above any matching `*` entries so it wins:

```bash
elif [[ "${TEST_CONFIG}" == *<name>* ]]; then
  python test/run_test.py --include test_ops -k 'my_test_name' --verbose
```

Anything you can run on a worker goes here: `python ...`, `pytest ...`,
benchmark scripts, `nvidia-smi`, `compute-sanitizer python ...`. There's
no requirement to run the test suite at all.

### 4. Strip workflows that fire on every PR

`pull.yml`, `lint.yml`, `lint-bc.yml` fire on every PR with no path
filter. On a one-off PR they burn ~100+ irrelevant jobs.

**Recommended**: remove the `pull_request:` trigger entirely from each
file:

```yaml
on:
  pull_request:           # delete these three lines
    branches-ignore:      #
      - nightly           #
  push:
    branches:
      - main
      ...
```

leaving:

```yaml
on:
  push:
    branches:
      - main
      ...
```

**About `paths: [.github/workflows/<self>.yml]`** — this idiom is
common in PyTorch (see `b200-distributed.yml`, `h100-distributed.yml`)
and means *"run this workflow on PRs that modify this file."* It's
correct for "fire when the workflow's config changes" but does **not**
suppress unrelated PR pushes from triggering the workflow when your PR
also modifies that workflow file. So as a *suppression* technique on a
debug PR, `paths: [<self>]` doesn't work — your debug PR keeps
modifying the file, retriggering on every push.

If you want a paths-based suppression that's still trivially reversible
(rather than removing the trigger), use a sentinel that won't ever match:

```yaml
pull_request:
  paths:
    - __never_match__
```

Other small workflows (`auto_request_review.yml`, `nitpicker.yml`,
`check_mergeability_ghstack.yml` for ghstack PRs) cost effectively
nothing — leave them alone.

### 5. Push the PR

```bash
git checkout -b <name>
git add .github/workflows/<name>.yml \
        .github/workflows/pull.yml \
        .github/workflows/lint.yml \
        .github/workflows/lint-bc.yml \
        .ci/pytorch/test.sh
git commit -m "CI workflow for <name>"
git push -u fork <name>
```

If the push is rejected with `workflows scope may be required`, refresh
your token's scopes:

```bash
gh auth refresh -s workflow
```

Open the PR as **draft** and label clearly that it's not for merge.

### 6. Read the logs

```bash
gh pr view <pr> --repo pytorch/pytorch --json statusCheckRollup \
  -q '.statusCheckRollup[] | select(.workflowName == "<name>")'

# Once the test job completes:
gh api repos/pytorch/pytorch/actions/jobs/<test_job_id>/logs > /tmp/run.log
```

The job's overall conclusion is whatever exit code your last command
produced. If you're catching errors with `|| true`, that conclusion is
meaningless — grep the log for actual failures.

## When chasing flaky tests specifically

Flakes that depend on accumulated state (CUDA context, allocator pools,
leaked tensors from earlier tests) are special. A few extra rules:

- **Don't run the failing test in isolation with `-k <name>`.** It almost
  always passes alone. Replicate the original failing command verbatim
  (e.g. `--include test_ops --shard 1 2`).
- **Loop the command 3-5 times** with `|| true` between iterations, so
  later attempts still run after one trips the bug:
  ```bash
  for i in 1 2 3; do
    python test/run_test.py --include test_ops --shard 1 2 || true
  done
  ```
- **CUDA device-side asserts cascade.** The first failing test trips an
  assertion; subsequent tests fail in confusing places (often
  `torch.cuda.get_rng_state()` inside Dynamo's `_fn` decorator, or in
  whatever `setUp()` does first). Find the first assert in the log, not
  the cascade.
- **Sample-input index in the log is reproducible-in-context, not in
  isolation.** `PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=N python test/test_ops.py
  ...` won't always reproduce on its own.
