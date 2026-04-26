"""
Enumerate all aten ops reachable through OpInfo, query sharding/decomp support,
and optionally verify Replicate DTensor execution.

For each OpInfo entry (all variants, all samples), runs under TorchDispatchMode
to capture every aten op hit. Then for each aten op, reports:
  - Sharding path: single_dim, op_strategy, decomp, or none
  - Decomp detail: explicit (in decomposition_table), cia (_can_decompose), or none
  - Replicate test: whether running with Replicate DTensors succeeds

Usage:
    python enumerate_opinfo_aten_ops.py --test-replicate --csv aten_op_coverage.csv
    python enumerate_opinfo_aten_ops.py --op nanmean --test-replicate
"""

import argparse
import csv
import sys
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Replicate
from torch.distributed.tensor._decompositions import DecompShardingStrategy
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import SampleInput

# Tensor constructors that don't take DTensor inputs and don't produce DTensors.
# Everything else arguably could have sharding support.
NEVER_NEED_COVERAGE = {
    "aten.empty.memory_format",
    "aten.rand.default",
    "aten.scalar_tensor.default",
}


def is_never_need_coverage(op_str: str) -> bool:
    return op_str in NEVER_NEED_COVERAGE


class _CaptureAllAtenOps(torch.utils._python_dispatch.TorchDispatchMode):
    """Capture every aten op (with args) during execution."""

    def __init__(self):
        self.ops: list[tuple] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.namespace == "aten":
            self.ops.append((func, args, kwargs))
        return func(*args, **kwargs)


def run_sample(op_callable, sample: SampleInput):
    if isinstance(sample.input, torch.Tensor):
        return op_callable(sample.input, *sample.args, **sample.kwargs)
    else:
        return op_callable(*sample.input, *sample.args, **sample.kwargs)


def dtensorify_sample(sample: SampleInput, mesh: DeviceMesh):
    """Replace all tensors in a SampleInput with Replicate DTensors."""

    def _to_dt(x):
        if isinstance(x, torch.Tensor):
            return distribute_tensor(x.clone().detach(), mesh, (Replicate(),))
        return x

    new_input = pytree.tree_map(_to_dt, sample.input)
    new_args = pytree.tree_map(_to_dt, sample.args)
    new_kwargs = pytree.tree_map(_to_dt, sample.kwargs)
    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def query_sharding_support(aten_op):
    """Return which sharding path an aten op uses, or None."""
    propagator = DTensor._op_dispatcher.sharding_propagator
    if aten_op in propagator.op_single_dim_strategy_funcs:
        return "single_dim"
    if aten_op in propagator.op_strategy_funcs:
        return "op_strategy"
    if DecompShardingStrategy.has_decomp(aten_op):
        return "decomp"
    return None


def query_decomp_detail(aten_op):
    """Return decomp source: 'explicit', 'cia', 'explicit+cia', or None."""
    from torch._decomp import decomposition_table

    in_table = aten_op in decomposition_table
    is_cia = aten_op._can_decompose()
    if in_table and is_cia:
        return "explicit+cia"
    elif in_table:
        return "explicit"
    elif is_cia:
        return "cia"
    return None


def process_opinfo(opinfo, device, dtype, max_samples, test_replicate, mesh):
    """Run samples for one OpInfo, capture all aten ops and replicate results."""
    all_aten_ops = set()
    replicate_ok = set()
    replicate_fail = {}

    try:
        samples = list(opinfo.sample_inputs(device, dtype))
    except Exception:
        return all_aten_ops, replicate_ok, replicate_fail

    for sample in (samples[:max_samples] if max_samples else samples):
        with _CaptureAllAtenOps() as capture:
            try:
                run_sample(opinfo.op, sample)
            except Exception:
                continue

        for func, _, _ in capture.ops:
            all_aten_ops.add(func)

        if test_replicate and mesh is not None:
            try:
                dt_sample = dtensorify_sample(sample, mesh)
                run_sample(opinfo.op, dt_sample)
                for func, _, _ in capture.ops:
                    replicate_ok.add(func)
            except Exception as e:
                err_str = str(e).split("\n")[0][:120]
                for func, _, _ in capture.ops:
                    if func not in replicate_ok and func not in replicate_fail:
                        replicate_fail[func] = err_str

    return all_aten_ops, replicate_ok, replicate_fail


def print_summary(
    aten_sharding,
    aten_decomp_detail,
    aten_to_opinfos,
    aten_replicate_ok,
    aten_replicate_fail,
    num_opinfos,
    test_replicate,
):
    total = len(aten_sharding)

    # Sharding path breakdown
    path_counts = defaultdict(int)
    for v in aten_sharding.values():
        path_counts[v or "(none)"] += 1

    # Decomp detail breakdown
    decomp_detail_counts = defaultdict(int)
    for op_str, sharding in aten_sharding.items():
        if sharding == "decomp":
            detail = aten_decomp_detail.get(op_str) or "(unknown)"
            decomp_detail_counts[detail] += 1

    # Replicate verification for decomp ops
    decomp_verified = 0
    decomp_failed = 0
    if test_replicate:
        for op_str, sharding in aten_sharding.items():
            if sharding != "decomp":
                continue
            ok = len(aten_replicate_ok.get(op_str, set()))
            fail = len(aten_replicate_fail.get(op_str, {}))
            if ok > 0:
                decomp_verified += 1
            elif fail > 0:
                decomp_failed += 1

    # Never-need-coverage count
    never_need = sum(1 for op_str in aten_sharding if is_never_need_coverage(op_str))
    no_sharding = path_counts["(none)"]
    actually_missing = sum(
        1 for op_str in aten_sharding
        if aten_sharding[op_str] is None and not is_never_need_coverage(op_str)
    )

    out = sys.stderr
    print(f"\n{'='*80}", file=out)
    print(f"OpInfo entries processed:  {num_opinfos}", file=out)
    print(f"Unique aten ops found:     {total}", file=out)
    print(f"", file=out)
    print(f"Sharding path breakdown:", file=out)
    print(f"  single_dim               {path_counts['single_dim']}", file=out)
    print(f"  op_strategy              {path_counts['op_strategy']}", file=out)
    print(f"  decomp                   {path_counts['decomp']}", file=out)
    print(f"  (none)                   {path_counts['(none)']}", file=out)
    print(f"", file=out)
    print(f"Decomp detail (of {path_counts['decomp']} decomp ops):", file=out)
    for detail, count in sorted(decomp_detail_counts.items(), key=lambda x: -x[1]):
        print(f"  {detail:<20}     {count}", file=out)
    print(f"", file=out)

    if test_replicate:
        print(f"Decomp Replicate verification:", file=out)
        print(f"  Confirmed working        {decomp_verified}", file=out)
        print(f"  Fails at runtime         {decomp_failed}", file=out)
        print(f"", file=out)

        verified_total = (
            path_counts["single_dim"]
            + path_counts["op_strategy"]
            + decomp_verified
        )
        relevant = total - never_need
        print(f"Coverage summary:", file=out)
        print(f"  Claimed support          {path_counts['single_dim'] + path_counts['op_strategy'] + path_counts['decomp']}  (single_dim + op_strategy + decomp)", file=out)
        print(f"  Verified working         {verified_total}  (single_dim + op_strategy + decomp_verified)", file=out)
        print(f"  No sharding              {no_sharding}", file=out)
        print(f"    Never need coverage    {never_need}", file=out)
        print(f"    Actually missing       {actually_missing}", file=out)
        print(f"  Coverage rate            {verified_total}/{relevant} = {100*verified_total/relevant:.1f}%  (verified / relevant ops)", file=out)
    else:
        print(f"No sharding:", file=out)
        print(f"  Never need coverage      {never_need}", file=out)
        print(f"  Actually missing         {actually_missing}", file=out)

    print(f"{'='*80}", file=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", type=str, default=None, help="Filter to a single op name")
    parser.add_argument("--test-replicate", action="store_true", help="Test Replicate DTensor execution")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per OpInfo (0=unlimited)")
    args = parser.parse_args()

    device = "cpu"
    dtype = torch.float32
    world_size = 2

    dist.init_process_group("fake", rank=0, world_size=world_size)
    mesh = DeviceMesh(device, list(range(world_size)))

    opinfos = [op for op in op_db if args.op is None or op.name == args.op]
    print(f"Processing {len(opinfos)} OpInfo entries...", file=sys.stderr)

    aten_to_opinfos: dict[str, set[str]] = defaultdict(set)
    aten_sharding: dict[str, str | None] = {}
    aten_decomp_detail: dict[str, str | None] = {}
    aten_replicate_ok: dict[str, set[str]] = defaultdict(set)
    aten_replicate_fail: dict[str, dict[str, str]] = defaultdict(dict)

    for i, opinfo in enumerate(opinfos):
        variant = opinfo.variant_test_name
        label = f"{opinfo.name}({variant})" if variant else opinfo.name
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(opinfos)}] {label}", file=sys.stderr)

        all_ops, rep_ok, rep_fail = process_opinfo(
            opinfo, device, dtype,
            max_samples=args.max_samples,
            test_replicate=args.test_replicate,
            mesh=mesh,
        )

        for aten_op in all_ops:
            op_str = str(aten_op)
            aten_to_opinfos[op_str].add(label)
            if op_str not in aten_sharding:
                aten_sharding[op_str] = query_sharding_support(aten_op)
            if op_str not in aten_decomp_detail:
                aten_decomp_detail[op_str] = query_decomp_detail(aten_op)
            if aten_op in rep_ok:
                aten_replicate_ok[op_str].add(label)
            if aten_op in rep_fail:
                aten_replicate_fail[op_str][label] = rep_fail[aten_op]

    # Summary
    print_summary(
        aten_sharding, aten_decomp_detail, aten_to_opinfos,
        aten_replicate_ok, aten_replicate_fail,
        num_opinfos=len(opinfos),
        test_replicate=args.test_replicate,
    )

    # Per-op table to stdout
    header = f"{'aten op':<55} {'sharding':<12} {'decomp_detail':<15} {'#opinfos':<10}"
    if args.test_replicate:
        header += f" {'repl_ok':<10} {'repl_fail':<10}"
    header += " opinfo_names"
    print(header)
    print("-" * len(header))

    for op_str in sorted(aten_sharding.keys()):
        sharding = aten_sharding[op_str] or "-"
        decomp_detail = aten_decomp_detail.get(op_str) or "-"
        n_opinfos = len(aten_to_opinfos[op_str])
        opinfo_names = ", ".join(sorted(aten_to_opinfos[op_str]))
        line = f"{op_str:<55} {sharding:<12} {decomp_detail:<15} {n_opinfos:<10}"
        if args.test_replicate:
            n_ok = len(aten_replicate_ok.get(op_str, set()))
            n_fail = len(aten_replicate_fail.get(op_str, {}))
            line += f" {n_ok:<10} {n_fail:<10}"
        line += f" {opinfo_names}"
        print(line)

    # CSV output
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            cols = [
                "aten_op", "sharding_path", "decomp_detail",
                "never_need_coverage", "num_opinfos", "opinfo_names",
            ]
            if args.test_replicate:
                cols += ["replicate_ok_count", "replicate_fail_count", "replicate_fail_errors"]
            writer.writerow(cols)
            for op_str in sorted(aten_sharding.keys()):
                row = [
                    op_str,
                    aten_sharding[op_str] or "",
                    aten_decomp_detail.get(op_str) or "",
                    is_never_need_coverage(op_str),
                    len(aten_to_opinfos[op_str]),
                    "; ".join(sorted(aten_to_opinfos[op_str])),
                ]
                if args.test_replicate:
                    n_ok = len(aten_replicate_ok.get(op_str, set()))
                    fails = aten_replicate_fail.get(op_str, {})
                    fail_summary = "; ".join(f"{k}: {v}" for k, v in sorted(fails.items())[:5])
                    row += [n_ok, len(fails), fail_summary]
                writer.writerow(row)
        print(f"\nCSV written to {args.csv}", file=sys.stderr)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
