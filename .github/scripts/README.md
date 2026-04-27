# pytorch/.github

> NOTE: This README contains information for the `.github` directory but cannot be located there because it will overwrite the
repo README.

This directory contains workflows and scripts to support our CI infrastructure that runs on GitHub Actions.

## Workflows

- Pull CI (`pull.yml`) is run on PRs and on main.
- Trunk CI (`trunk.yml`) is run on trunk to validate incoming commits. Trunk jobs are usually more expensive to run so we do not run them on PRs unless specified.
- Scheduled CI (`periodic.yml`) is a subset of trunk CI that is run every few hours on main.
- Binary CI is run to package binaries for distribution for all platforms.

## Binary build workflows

Binary build workflows live under `.github/workflows/generated-*.yml` and are
hand-edited. The GPU/arch/Python matrix shared across them is defined in
`.github/scripts/generate_binary_build_matrix.py` and consumed at runtime via
`--runtime-matrix <os>` (emits `configs=<json>` to `$GITHUB_OUTPUT`). Adding a
new CUDA/ROCm/Python version to that script propagates to the workflows on the
next run with no YAML change.

#### ciflow (trunk)

The label `ciflow/trunk` can be used to run `trunk` only workflows. This is especially useful if trying to re-land a PR that was
reverted for failing a `non-default` workflow.

## Infra

Currently most of our self hosted runners are hosted on AWS, for a comprehensive list of available runner types you
can reference `.github/scale-config.yml`.

Exceptions to AWS for self hosted:
* ROCM runners

### Adding new runner types

New runner types can be added by committing changes to `.github/scale-config.yml`. Example: https://github.com/pytorch/pytorch/pull/70474

> NOTE: New runner types can only be used once the changes to `.github/scale-config.yml` have made their way into the default branch

