"""
Analyze OpInfo coverage vs the overridable torch API.

Compares op_db against get_testing_overrides() to find gaps.

Usage: python opinfo_coverage_analysis.py [-o FILE]
"""

import argparse
import sys

import torch
from torch.testing._internal.common_methods_invocations import op_db

# Prefixes for non-compute functions excluded from gap analysis:
# predicates, backend-specific, symbolic, quantization, initializers.
NON_COMPUTE_PREFIXES = (
    "is_", "q_", "quantize", "quantized_", "fbgemm_",
    "miopen_", "sym_", "nn.init.",
)

# alias -> primary OpInfo name (from the `aliases` field on OpInfo entries)
KNOWN_ALIASES = {
    "absolute": "abs", "arccos": "acos", "arccosh": "acosh",
    "arcsin": "asin", "arcsinh": "asinh", "arctan": "atan",
    "arctan2": "atan2", "arctanh": "atanh", "clip": "clamp",
    "concat": "cat", "concatenate": "cat", "det": "linalg.det",
    "divide": "div", "fix": "trunc", "ger": "outer",
    "greater": "gt", "greater_equal": "ge", "inverse": "linalg.inv",
    "less": "lt", "less_equal": "le", "matrix_power": "linalg.matrix_power",
    "moveaxis": "movedim", "multiply": "mul", "negative": "neg",
    "not_equal": "ne", "orgqr": "linalg.householder_product",
    "row_stack": "vstack", "slogdet": "linalg.slogdet",
    "subtract": "sub", "swapaxes": "transpose", "swapdims": "transpose",
}


def get_override_names():
    """Short names of overridable torch.* functions (excludes Tensor methods and dunders)."""
    names = set()
    for fn in torch.overrides.get_testing_overrides():
        qname = torch.overrides.resolve_name(fn)
        if not qname or qname.startswith("torch.Tensor.") or "__" in qname:
            continue
        short = qname.removeprefix("torch.").removeprefix("functional.")
        names.add(short)
    return names


def is_alias(name, opinfo_names):
    if name in KNOWN_ALIASES:
        return KNOWN_ALIASES[name]
    if name.startswith("special."):
        base = name.removeprefix("special.")
        if base in opinfo_names:
            return base
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()
    out = open(args.output, "w") if args.output else sys.stdout

    override_names = get_override_names()
    opinfo_names = {op.name for op in op_db}

    non_compute = {n for n in override_names if n.startswith(NON_COMPUTE_PREFIXES)}
    compute = override_names - non_compute
    covered = compute & opinfo_names
    uncovered = compute - opinfo_names
    aliases = {n for n in uncovered if is_alias(n, opinfo_names)}
    true_gap = uncovered - aliases

    print(f"Overrides: {len(override_names)}  Compute: {len(compute)}  "
          f"op_db: {len(opinfo_names)}", file=out)
    print(f"Covered: {len(covered)}  Uncovered: {len(uncovered)} "
          f"(aliases: {len(aliases)}, true gap: {len(true_gap)})", file=out)
    print(f"Raw: {len(covered)*100/len(compute):.1f}%  "
          f"Alias-adjusted: {(len(covered)+len(aliases))*100/len(compute):.1f}%", file=out)

    for label, ops in [
        ("COVERED", sorted(covered)),
        ("TRUE GAP", sorted(true_gap)),
        ("ALIASES (covered under primary name)", sorted(aliases)),
        ("EXCLUDED non-compute", sorted(non_compute)),
    ]:
        print(f"\n{label} ({len(ops)})", file=out)
        for name in ops:
            alias_of = is_alias(name, opinfo_names)
            suffix = f"  -> {alias_of}" if alias_of else ""
            print(f"  {name}{suffix}", file=out)

    if args.output:
        out.close()
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
