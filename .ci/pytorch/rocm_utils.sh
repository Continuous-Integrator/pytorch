#!/bin/bash
# ROCm-specific utility functions shared across CI scripts

# Build rocm-composable-kernel (ck4inductor) wheel
# Usage: build_rocm_ck_wheel <output_dir>
build_rocm_ck_wheel() {
  local output_dir="${1:?Output directory required}"

  echo "Building rocm-composable-kernel (ck4inductor) wheel at $(date)"

  local ck_commit
  ck_commit=$(cat .ci/docker/ci_commit_pins/rocm-composable-kernel.txt)
  echo "CK commit: $ck_commit"

  git clone --depth 1 https://github.com/ROCm/composable_kernel.git /tmp/ck
  pushd /tmp/ck || return 1
  git fetch --depth 1 origin "$ck_commit"
  git checkout "$ck_commit"
  python -m build --wheel --no-isolation --outdir "$output_dir"
  popd || return 1
  rm -rf /tmp/ck

  echo "Finished building rocm-composable-kernel (ck4inductor) wheel at $(date)"
}
