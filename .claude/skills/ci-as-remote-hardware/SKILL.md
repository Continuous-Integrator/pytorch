---
name: ci-as-remote-hardware
description: Use PyTorch CI as remote hardware - set up a minimal CI workflow that runs an arbitrary command on a specific runner type (g5/A10G, MPS Mac, ROCm, etc) you don't have locally. Use when validating an MPS kernel without a Mac, debugging a CUDA kernel on a specific GPU SKU, reproducing a CI-only failure, or running anything that needs hardware/toolchain you can't install locally.
---

# Use PyTorch CI as Remote Hardware

Procedure: [`.github/RUNNING_ON_CI_HARDWARE.md`](../../../.github/RUNNING_ON_CI_HARDWARE.md). Read it first.

## Things you'll trip over if you don't read carefully

- **Check for an existing `ciflow/<label>/*` tag first.** Many existing
  workflows accept opt-in via tag (push `ciflow/h100-distributed/<pr-num>`).
  Don't write a new workflow when an existing one would do — the doc's
  "First: can you reuse" section lists tag families.

- **`paths: [<self>]` does NOT suppress your debug PR's reruns.** It
  fires whenever the file changes, which your debug PR keeps doing.
  Either delete the `pull_request:` trigger entirely or use
  `paths: [__never_match__]`.

- **`SUCCESS` after `|| true` is meaningless.** When you wrap a command
  in `|| true` to keep a loop running, the job conclusion no longer
  reflects whether the test passed. Always grep the log for actual
  failures rather than trusting the overall status.

- **Pushing `.github/workflows/*` requires `workflow` scope.** If the
  push is rejected with "workflows scope may be required," run
  `gh auth refresh -s workflow`. The user may need to do this themselves.

- **Python 3.10 forbids backslashes inside f-string expressions.**
  When writing shell heredocs that pipe into `python3 -c`, use
  `%`-formatting (`"... %s ..." % name`) — not
  `f"... {x.replace(...)} ..."` — or you'll waste an iteration debugging
  syntax errors.

- **Don't filter a flaky test to `-k <name>` and expect it to repro.**
  Flaky tests usually depend on state from earlier tests in the same
  shard. Replicate the original failing command verbatim. The doc's
  "When chasing flaky tests specifically" section has the full rules.
