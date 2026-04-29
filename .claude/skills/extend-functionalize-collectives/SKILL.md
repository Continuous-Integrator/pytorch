---
name: extend-functionalize-collectives
description: Add support for a new inplace c10d collective (e.g. broadcast_, _allgather_base_, _reduce_scatter_base_, allgather_, reduce_scatter_, alltoall_, alltoall_base_, reduce_, send, recv_) to the FX-level functionalize-collectives pass in torch/fx/passes/regional_inductor.py. Use when the user wants to extend regional_inductor to handle distributed ops beyond all_reduce, when make_fx-traced graphs containing dist.* APIs hit "Tried to deepcopy object __torch__.torch.classes.c10d.ProcessGroup" errors for ops other than allreduce_, or when the user mentions extending _functionalize_inplace_collectives, _inplace_c10d_rewrites, _emit_collective_chain, _resolve_process_group_name, _resolve_reduce_op_str, or _rewrite_*_ helpers.
---

# Extend functionalize-collectives pass

This skill teaches Claude how to add a new inplace c10d collective to the
functionalize-collectives FX pass at
`torch/fx/passes/regional_inductor.py:_functionalize_inplace_collectives`.

The pass mirrors what Dynamo's `traceable_collective_remaps` does at trace
time, but operates on graphs already produced by `make_fx`. It rewrites
opaque `torch.ops.c10d.{op}_` calls (which carry non-deepcopiable
ProcessGroup/ReduceOp torchbind ScriptObjects) into the
`_c10d_functional.{op}` + `wait_tensor` + `aten.copy_` form, so downstream
consumers like `standalone_compile` can deepcopy the GraphModule.

## When to use this skill

- The user asks to support a new collective in `regional_inductor`, e.g.
  "add `broadcast_` support", "make `_allgather_base_` work after make_fx".
- A `make_fx`-traced graph crashes downstream with
  `Tried to deepcopy object __torch__.torch.classes.c10d.ProcessGroup` for
  an op other than `allreduce_`.
- The user mentions any of the helpers: `_functionalize_inplace_collectives`,
  `_inplace_c10d_rewrites`, `_emit_collective_chain`,
  `_resolve_process_group_name`, `_resolve_reduce_op_str`,
  `_redirect_tensors_work_uses`, `_redirect_tensor_work_uses`, or wants
  to add a new `_rewrite_<op>_`.

## Architecture

The pass is built from small composable pieces. Per-op rewrites only need to
glue them together — they don't reinvent any plumbing.

```
_functionalize_inplace_collectives(gm)              # outer driver
    │
    ├── _inplace_c10d_rewrites()                    # dispatch dict
    │
    ├── _rewrite_<op>_(gm, node)                    # per-op rewrite
    │     ├── _resolve_process_group_name(gm, arg)  # PG  -> str
    │     ├── _resolve_reduce_op_str(gm, arg)       # ReduceOp -> "sum"/...
    │     ├── _emit_collective_chain(...)           # functional + wait + copy_
    │     └── _redirect_{tensors,tensor}_work_uses  # depending on schema
    │
    └── final cleanup: erase node, dead get_attr, drop _torchbind_obj* attrs
```

Three things are shared across all rewrites:

1. **Arg resolution.** `_resolve_torchbind_arg` unboxes `get_attr` nodes;
   `_resolve_process_group_name` returns `pg.group_name`;
   `_resolve_reduce_op_str` converts a `ReduceOp` ScriptObject (or enum) to
   `"sum"`/`"avg"`/etc.
2. **Chain construction.** `_emit_collective_chain(gm, before, input_t,
   output_t, target, extra_args, group_name, custom)` inserts
   `target(input, *extra, group_name)` -> `wait_tensor` -> `aten.copy_(output,
   wait)` and propagates `meta["val"]` (using `output_t`'s fake tensor) and
   `meta["custom"]`. Returns the wait node.
3. **Use redirection.** The per-op rewrite picks the helper matching the
   output schema (see table below).

## Output schemas

Inplace c10d ops fall into three groups. Pick the matching `_redirect_*`
helper for the new op:

| Output schema       | Example ops                                                                | Use redirector                  |
|---------------------|----------------------------------------------------------------------------|---------------------------------|
| `(Tensor[], Work)`  | `allreduce_`, `broadcast_`, `reduce_scatter_`, `alltoall_`                  | `_redirect_tensors_work_uses`   |
| `(Tensor, Work)`    | `_allgather_base_`, `_reduce_scatter_base_`                                | `_redirect_tensor_work_uses`    |
| `Work` (no tensor)  | `alltoall_base_`, `reduce_`, `send`, `recv_`                                | (none — the op is pure side-effect; `aten.copy_` already updates the input/output tensor in place) |

For `Work`-only ops there is no tensor `getitem(node, 0)` to redirect; the
inplace `aten.copy_(output_t, wait)` emitted by `_emit_collective_chain`
already takes care of the output mutation. You only need to erase the
optional `getitem(node, 0)` for the work handle, if any. If you find that
schema in practice, add a `_redirect_work_only_uses` helper following the
same pattern.

## Common arg patterns

Determining inputs/outputs and extra args by op family:

| c10d op                  | Input tensor(s)   | Output tensor(s)   | Functional target                                  | Extra args                                                |
|--------------------------|-------------------|--------------------|----------------------------------------------------|-----------------------------------------------------------|
| `allreduce_`             | `tensors[i]`      | `tensors[i]` (same) | `_c10d_functional.all_reduce`                      | `(op_str,)`                                               |
| `broadcast_`             | `tensors[i]`      | `tensors[i]` (same) | `_c10d_functional.broadcast`                       | `(root_rank,)`                                            |
| `_allgather_base_`       | `args[1]` (input) | `args[0]` (output) | `_c10d_functional.all_gather_into_tensor`          | `(group_size,)` from `pg.size()`                          |
| `_reduce_scatter_base_`  | `args[1]`         | `args[0]`          | `_c10d_functional.reduce_scatter_tensor`           | `(op_str, group_size)`                                    |
| `allgather_`             | per input list    | concat output[][]  | `_c10d_functional.all_gather_into_tensor` + slices | per-input group_size, then split output back              |
| `reduce_scatter_`        | concat input[][]  | per output list    | `_c10d_functional.reduce_scatter_tensor`           | `(op_str, group_size)` per element                        |

Look at the Python wrappers in `torch/distributed/_functional_collectives.py`
(e.g. `all_reduce_inplace`, `all_gather_tensor_inplace`,
`reduce_scatter_tensor_inplace`) for an authoritative source of how each
inplace API maps to the functional collective + wait + copy_ pattern. This
pass mirrors those wrappers at the FX level.

## How to add a new op

### Step 1 — Inspect the schema

```python
print(torch.ops.c10d.<op>_.default._schema)
```

Note positions of: tensor list / output / input args, ProcessGroup arg,
ReduceOp arg (if any), and the output tuple shape.

### Step 2 — Write `_rewrite_<op>_(gm, node)`

Two templates cover almost every case.

**Template A — `(Tensor[], Work)` schema** (per-tensor rewrite, e.g. `broadcast_`):

```python
def _rewrite_broadcast_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Schema: c10d::broadcast_(Tensor[] tensors, ProcessGroup pg,
    int root_rank, int root_tensor, bool async_op, int timeout=-1)
    -> (Tensor[], Work)
    """
    tensors = node.args[0]
    group_name = _resolve_process_group_name(gm, node.args[1])
    root_rank = node.args[2]
    custom = node.meta.get("custom")
    target = torch.ops._c10d_functional.broadcast.default
    waits = [
        _emit_collective_chain(gm, node, t, t, target, (root_rank,), group_name, custom)
        for t in tensors
    ]
    _redirect_tensors_work_uses(gm, node, waits)
```

**Template B — `(Tensor, Work)` schema with separate output/input** (e.g.
`_allgather_base_`, `_reduce_scatter_base_`):

```python
def _rewrite__allgather_base_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Schema: c10d::_allgather_base_(Tensor output, Tensor input,
    ProcessGroup pg, bool async_op, int timeout=-1) -> (Tensor, Work)
    """
    import torch.distributed as dist

    output_t, input_t = node.args[0], node.args[1]
    group_name = _resolve_process_group_name(gm, node.args[2])
    pg = _resolve_torchbind_arg(gm, node.args[2])
    if isinstance(pg, torch.ScriptObject):
        pg = dist.ProcessGroup.unbox(pg)
    group_size = pg.size()
    custom = node.meta.get("custom")
    target = torch.ops._c10d_functional.all_gather_into_tensor.default
    wait = _emit_collective_chain(
        gm, node, input_t, output_t, target, (group_size,), group_name, custom
    )
    _redirect_tensor_work_uses(gm, node, wait)
```

Notes:
- `_emit_collective_chain` copies `output_t.meta["val"]` onto every new
  node. That works for ops where the functional collective produces a tensor
  with the same shape/dtype as `output_t`. For most c10d collectives this
  holds (the inplace c10d op was already storing into `output_t`, and the
  functional collective produces the same layout). If a future op needs a
  different intermediate shape, materialize the val under the input's
  fake_mode and set `meta["val"]` manually instead of relying on the helper.

### Step 3 — Register in `_inplace_c10d_rewrites`

```python
def _inplace_c10d_rewrites() -> dict[Any, _InplaceCollectiveRewrite]:
    if not torch.distributed.is_available():
        return {}
    return {
        torch.ops.c10d.allreduce_.default: _rewrite_allreduce_,
        torch.ops.c10d.<op>_.default: _rewrite_<op>_,   # <-- new
    }
```

That is the entire wiring. The driver discovers the new rewrite, calls it,
erases the rewritten node, and runs cleanup automatically.

### Step 4 — Add a test

Add a case to `test/distributed/test_regional_inductor_collectives.py`. The
file initialises a `fake` backend with `FakeStore`, so no real distributed
runtime is needed.

```python
def test_functionalize_inplace_<op>(self):
    def f(t):
        t = t.clone()
        dist.<op>(t, ...)        # the dist.* API that lowers to c10d.<op>_
        return t + 1

    gm = make_fx(f)(torch.ones(...))
    FileCheck().check("c10d.<op>_").run(str(gm.graph))

    _functionalize_inplace_collectives(gm)

    FileCheck() \
        .check_not("c10d.<op>_") \
        .check("_c10d_functional.<op>") \
        .check("wait_tensor") \
        .check("aten.copy_") \
        .run(str(gm.graph))

    copy.deepcopy(gm)            # must succeed: no torchbind attrs left
```

Run:
```bash
python test/distributed/test_regional_inductor_collectives.py
```

### Step 5 — Lint

```bash
lintrunner torch/fx/passes/regional_inductor.py test/distributed/test_regional_inductor_collectives.py
lintrunner -a torch/fx/passes/regional_inductor.py test/distributed/test_regional_inductor_collectives.py  # autofix
```

## Common pitfalls

- **Reading `pg.group_name` on the raw ScriptObject fails.** The torchbind
  `ProcessGroup` exposes only `__init__`. Use `_resolve_process_group_name`
  (or `dist.ProcessGroup.unbox(pg)` if you need other PG state like
  `pg.size()`).
- **Treating `dist.ReduceOp.SUM` and the torchbind `ReduceOp` as
  interchangeable.** They are not. Always go through `_resolve_reduce_op_str`,
  which handles both forms.
- **Forgetting `meta["val"]` on the new nodes** breaks the partitioner with
  `Partition is bad because non fake tensor value is seen ...`.
  `_emit_collective_chain` propagates it for you; only override when the
  output shape genuinely differs from `output_t`.
- **Forgetting `meta["custom"]`** drops the `compile_with_inductor`
  annotation, so the new functional ops fall outside any inductor region
  and the partitioner may split awkwardly. `_emit_collective_chain` carries
  it forward automatically.
- **Not erasing the original c10d node.** `eliminate_dead_code` treats
  inplace c10d ops as impure (they are), so it will not remove them. The
  driver erases via `gm.graph.erase_node(node)` after the rewrite runs.
- **Leaving torchbind attrs on the GraphModule.** Even after the `get_attr`
  node is gone, `gm._torchbind_obj0` itself remains in `gm.__dict__` and
  re-triggers the deepcopy crash. The driver cleans these up; verify with:
  ```python
  assert not [k for k in gm.__dict__ if k.startswith("_torchbind_obj")]
  ```

## Reference

- Pass driver, helpers, `_rewrite_allreduce_`:
  `torch/fx/passes/regional_inductor.py` — search for
  `--- functionalize-collectives pass ---`.
- Inplace c10d schemas: `torch/csrc/distributed/c10d/Ops.cpp`.
- Functional collective wrappers (best source for arg derivation):
  `torch/distributed/_functional_collectives.py` — see
  `all_reduce_inplace`, `all_gather_tensor_inplace`,
  `reduce_scatter_tensor_inplace`, `all_to_all_inplace`.
- Dynamo's parallel rewrite at trace time: search for
  `traceable_collective_remaps` in
  `torch/distributed/_functional_collectives.py` and
  `CollectiveFunctionRewriteVariable` in
  `torch/_dynamo/variables/functions.py`.
- Existing test file: `test/distributed/test_regional_inductor_collectives.py`.
