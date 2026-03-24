"""Dual-path FX CodeGen with zero-overhead profiler support.

Generates two forward paths from FX graph IR:
  - ``_forward_impl``: clean code, zero profiler overhead
  - ``_forward_profiled``: per-node ``_RecordFunctionFast`` wrapping

At runtime, ``forward()`` dispatches based on
``torch.autograd.profiler._is_profiler_enabled``.

Usage::

    from torch.fx.profiler_codegen import ProfilerCodeGen

    gm = torch.export.export(model, args).module()
    gm.graph.set_codegen(ProfilerCodeGen())
    gm.recompile()

    # Without profiler: runs _forward_impl (zero overhead)
    output = gm(input)

    # With profiler: runs _forward_profiled (per-op RecordFunctionFast)
    with torch.profiler.profile() as prof:
        output = gm(input)
"""

import enum
import re
import types
import typing
from typing import Any

import torch
import torch.fx
from torch._library.opaque_object import get_opaque_obj_repr, is_opaque_value_type
from torch.fx.graph import (
    _counter_regexp,
    _custom_builtins,
    _format_target,
    _get_qualified_name,
    _is_from_torch,
    _Namespace,
    _origin_type_map,
    _type_repr,
    CodeGen,
    inplace_methods,
    magic_methods,
    PythonCode,
)
from torch.fx.node import Node


class ProfilerCodeGen(CodeGen):
    """CodeGen that emits _forward_impl, _forward_profiled, and a forward dispatcher."""

    def __init__(self) -> None:
        super().__init__()

    def _gen_python_code(
        self,
        nodes,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        """Generate dual-path source from FX graph nodes.

        Maintains two body lists (``impl_body``, ``profiled_body``) and
        populates both in a single pass over the nodes.  The
        ``record_func`` parameter is ignored.
        """
        free_vars: list[str] = []
        impl_body: list[str] = []
        profiled_body: list[str] = []
        globals_: dict[str, Any] = {}
        wrapped_fns: dict[str, None] = {}
        maybe_return_annotation: list[str] = [""]

        def add_global(name_hint: str, obj: Any):
            if _is_from_torch(obj) and obj != torch.device:
                return _get_qualified_name(obj)
            global_name = namespace.create_name(name_hint, obj)
            if global_name in globals_:
                if globals_[global_name] != obj:
                    raise AssertionError(
                        f"Global name {global_name} already assigned to different object"
                    )
                return global_name
            globals_[global_name] = obj
            return global_name

        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o: Any):
            if o == ():
                return "()"
            typename = _type_repr(o)
            if isinstance(o, types.UnionType) and "|" in typename:
                args = [type_repr(arg) for arg in o.__args__]
                return "|".join(args)
            if origin_type := getattr(o, "__origin__", None):
                if isinstance(o, typing._GenericAlias):  # type: ignore[attr-defined]
                    origin_type = _origin_type_map.get(origin_type, origin_type)
                origin_typename = add_global(_type_repr(origin_type), origin_type)
                if hasattr(o, "__args__") and o.__args__:
                    args = [type_repr(arg) for arg in o.__args__]
                    return f"{origin_typename}[{','.join(args)}]"
                else:
                    return origin_typename
            return add_global(typename, o)

        def _get_repr(arg: Any) -> str:
            if isinstance(arg, Node):
                return repr(arg)
            elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
                qualified_name = _get_qualified_name(type(arg))
                global_name = add_global(qualified_name, type(arg))
                return f"{global_name}{repr(tuple(arg))}"
            elif isinstance(
                arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                qualified_name = _get_qualified_name(arg)
                global_name = add_global(qualified_name, arg)
                return f"{global_name}"
            elif isinstance(arg, enum.Enum):
                cls = arg.__class__
                clsname = add_global(cls.__name__, cls)
                return f"{clsname}.{arg.name}"
            elif isinstance(arg, torch.Tensor):
                size = list(arg.size())
                dtype = str(arg.dtype).split(".")[-1]
                return f"torch.Tensor(size={size}, dtype={dtype})"
            elif isinstance(arg, tuple):
                if len(arg) == 1:
                    return f"({_get_repr(arg[0])},)"
                else:
                    return "(" + ", ".join(_get_repr(a) for a in arg) + ")"
            elif isinstance(arg, list):
                return "[" + ", ".join(_get_repr(a) for a in arg) + "]"
            elif isinstance(arg, slice):
                return f"slice({_get_repr(arg.start)}, {_get_repr(arg.stop)}, {_get_repr(arg.step)})"
            elif is_opaque_value_type(type(arg)):
                obj_repr, opaque_types = get_opaque_obj_repr(arg)
                for n, t in opaque_types.items():
                    add_global(n, t)
                return obj_repr
            else:
                return repr(arg)

        def _format_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
            res = [_get_repr(a) for a in args]
            res.extend([f"{k} = {_get_repr(v)}" for k, v in kwargs.items()])
            return ", ".join(res)

        node_to_last_use: dict[Node, Node] = {}
        user_to_last_uses: dict[Node, list[Node]] = {}

        def register_last_uses(n: Node, user: Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            for input_node in node._input_nodes:
                register_last_uses(input_node, node)

        def delete_unused_values(user: Node):
            """Append cleanup to BOTH bodies."""
            if user.op == "placeholder":
                return
            if user.op == "output":
                impl_body.append("\n")
                profiled_body.append("\n")
                return
            nodes_to_delete = user_to_last_uses.get(user, [])
            if len(user.users.keys()) == 0:
                nodes_to_delete.append(user)
            if len(nodes_to_delete):
                to_delete_str = " = ".join(
                    [repr(n) for n in nodes_to_delete] + ["None"]
                )
                cleanup = f";  {to_delete_str}\n"
                impl_body.append(cleanup)
                profiled_body.append(cleanup)
            else:
                impl_body.append("\n")
                profiled_body.append("\n")

        def emit_node(node: Node):
            """Emit node code into impl_body (forked from base CodeGen)."""
            maybe_type_annotation = (
                "" if node.type is None else f" : {type_repr(node.type)}"
            )

            if node.op == "placeholder":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str, got {type(node.target)}"
                    )
                maybe_default_arg = (
                    "" if not node.args else f" = {_get_repr(node.args[0])}"
                )
                free_vars.append(
                    f"{node.target}{maybe_type_annotation}{maybe_default_arg}"
                )
                raw_name = node.target.replace("*", "")
                if raw_name != repr(node):
                    impl_body.append(f"{repr(node)} = {raw_name}\n")
                return
            elif node.op == "call_method":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for call_method, got {type(node.target)}"
                    )
                impl_body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}"
                    f"({_format_args(node.args[1:], node.kwargs)})"
                )
                return
            elif node.op == "call_function":
                if not callable(node.target):
                    raise AssertionError(
                        f"Expected node.target to be callable, got {type(node.target)}"
                    )
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in magic_methods
                ):
                    if not isinstance(node.args, tuple):
                        raise AssertionError(
                            f"Expected node.args to be tuple, got {type(node.args)}"
                        )
                    impl_body.append(
                        f"{repr(node)}{maybe_type_annotation} = "
                        f"{magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}"
                    )
                    return
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in inplace_methods
                ):
                    impl_body.append(
                        f"{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  "
                        f"{repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}"
                    )
                    return
                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                if (
                    global_name == "getattr"
                    and isinstance(node.args, tuple)
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                    and len(node.args) == 2
                ):
                    impl_body.append(
                        f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}"
                    )
                    return
                impl_body.append(
                    f"{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})"
                )
                if node.meta.get("is_wrapped", False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == "call_module":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for call_module, got {type(node.target)}"
                    )
                impl_body.append(
                    f"{repr(node)}{maybe_type_annotation} = "
                    f"{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})"
                )
                return
            elif node.op == "get_attr":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for get_attr, got {type(node.target)}"
                    )
                impl_body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}"
                )
                return
            elif node.op == "output":
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                desc = node.meta.get("desc", None) if expanded_def else None
                if desc is not None:
                    impl_body.append(
                        self._call_method_with_signature_check(
                            self.generate_output,
                            node.args[0],
                            descs=desc,
                        )
                    )
                else:
                    impl_body.append(
                        self._call_method_with_signature_check(
                            self.generate_output,
                            node.args[0],
                        )
                    )
                return
            raise NotImplementedError(f"node: {node.op} {node.target}")

        # Core loop: iterate nodes, emit into both bodies
        for i, node in enumerate(nodes):
            impl_body.append(f"# COUNTER: {i}\n")
            profiled_body.append(f"# COUNTER: {i}\n")

            is_recordable = node.op in (
                "call_function",
                "call_method",
                "call_module",
            )

            # Capture lines emitted by emit_node() into impl_body so we
            # can replay them (possibly indented) into profiled_body.
            body_len_before = len(impl_body)
            emit_node(node)
            emitted_lines = impl_body[body_len_before:]

            if is_recordable and emitted_lines:
                label = self._get_profiler_label(node).replace('"', '\\"')
                args_tuple = self._format_args_tuple(node)
                rf_var = f"_rf_{node.name}"
                profiled_body.append(
                    f"{rf_var} = torch._C._profiler._RecordFunctionFast("
                    f'"{label}", {args_tuple}); {rf_var}.__enter__()\n'
                )
                profiled_body.extend(emitted_lines)
                delete_unused_values(node)
                profiled_body.append(f"{rf_var}.__exit__(None, None, None)\n")
            else:
                profiled_body.extend(emitted_lines)
                delete_unused_values(node)

        if len(impl_body) == 0:
            impl_body.append("pass\n")
            profiled_body.append("pass\n")

        if len(wrapped_fns) > 0:
            wrap_name = add_global("wrap", torch.fx.wrap)
            wrap_stmts = "\n".join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ""

        if self._body_transformer:
            impl_body = self._body_transformer(impl_body)
            profiled_body = self._body_transformer(profiled_body)

        for name, value in self.additional_globals():
            add_global(name, value)

        # Assemble the three functions
        impl_prologue = self._gen_fn_def_with_name(
            "_forward_impl",
            free_vars,
            maybe_return_annotation[0],
            expanded_def=expanded_def,
        )
        impl_code, impl_lineno_map, impl_prologue_start = self._assemble_function(
            impl_prologue, impl_body, ""
        )

        profiled_prologue = self._gen_fn_def_with_name(
            "_forward_profiled",
            free_vars,
            maybe_return_annotation[0],
            expanded_def=expanded_def,
        )
        profiled_code, _, _ = self._assemble_function(
            profiled_prologue,
            profiled_body,
            "",
        )

        dispatcher_code = self._generate_dispatcher(free_vars)

        profiler_import = "import torch.autograd.profiler as _autograd_profiler"
        prefix = f"\n{wrap_stmts}\n\n{profiler_import}\n\n"
        combined = f"{prefix}{impl_code}\n\n{profiled_code}\n\n{dispatcher_code}\n"

        # Shift lineno_map keys to account for prefix lines before impl_code
        prefix_lines = prefix.count("\n")
        shifted_lineno_map = {k + prefix_lines: v for k, v in impl_lineno_map.items()}

        return PythonCode(
            combined,
            globals_,
            _lineno_map=shifted_lineno_map,
            _prologue_start=impl_prologue_start + prefix_lines,
        )

    def _gen_fn_def_with_name(
        self,
        func_name: str,
        free_vars: list[str],
        return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """Generate a ``def`` line with the given function name."""
        vars_copy = list(free_vars)
        if len(vars_copy) == 0 or vars_copy[0] != "self":
            vars_copy.insert(0, "self")
        if expanded_def:
            args_formatted = self._format_multiline_args(vars_copy)
            return f"def {func_name}(\n{args_formatted}){return_annotation}:"
        return f"def {func_name}({', '.join(vars_copy)}){return_annotation}:"

    def _assemble_function(
        self,
        prologue: str,
        body: list[str],
        wrap_stmts: str,
    ) -> tuple[str, dict[int, int | None], int]:
        """Assemble function source from prologue + body.

        Strips ``# COUNTER:`` comments and builds ``lineno_map``.
        """
        lineno_map: dict[int, int | None] = {}
        prologue_len = prologue.count("\n") + 1
        new_lines: list[str] = []
        cur_idx = None
        for line in "".join(body).split("\n"):
            counter = _counter_regexp.search(line)
            if counter is not None:
                cur_idx = int(counter.group(1))
            else:
                lineno_map[len(new_lines) + prologue_len] = cur_idx
                new_lines.append(line)

        code = "\n".join(new_lines).lstrip("\n")
        code = "\n".join("    " + line for line in code.split("\n"))

        fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
        prologue_start = wrap_stmts.count("\n") + 4
        return fn_code, lineno_map, prologue_start

    def _generate_dispatcher(self, free_vars: list[str]) -> str:
        """Generate ``forward()`` that dispatches on profiler state."""
        vars_copy = list(free_vars)
        if len(vars_copy) == 0 or vars_copy[0] != "self":
            vars_copy.insert(0, "self")
        params = ", ".join(vars_copy)
        # Strip type annotations for call sites (e.g. "input : torch.Tensor" -> "input")
        call_args = ", ".join(v.split(":")[0].split("=")[0].strip() for v in vars_copy)
        return (
            f"def forward({params}):\n"
            f"    # _is_profiler_enabled is a module-level bool — single attr read, zero overhead\n"
            f"    if _autograd_profiler._is_profiler_enabled:\n"
            f"        return _forward_profiled({call_args})\n"
            f"    return _forward_impl({call_args})\n"
        )

    def _get_profiler_label(self, node: Node) -> str:
        """Format: ``"node_name: op_name (file:line)"``."""
        op_name = self._get_op_name(node)
        source_loc = self._get_source_location(node)
        label = f"{node.name}: {op_name}"
        if source_loc:
            label = f"{label} ({source_loc})"
        return label

    def _get_op_name(self, node: Node) -> str:
        if node.op == "call_function":
            if hasattr(node.target, "__name__"):
                return node.target.__name__
            elif hasattr(node.target, "_name"):
                return node.target._name
            return str(node.target).split(".")[-1]
        return str(node.target)

    _SOURCE_LOC_PATTERN = re.compile(r'^File "(.+)", line (\d+), in (.+)$')

    def _get_source_location(self, node: Node) -> str:
        """Extract ``filename:lineno`` from the node's stack trace.

        Skips internal PyTorch and site-packages frames.
        """
        stack_trace = node.meta.get("stack_trace", None)
        if not stack_trace:
            return ""

        lines = stack_trace.strip().split("\n")
        for idx in range(len(lines) - 2, -1, -1):
            line = lines[idx].strip()
            match = self._SOURCE_LOC_PATTERN.match(line)
            if match:
                filepath = match.group(1)
                lineno = match.group(2)
                if "/torch/" in filepath or "/site-packages/" in filepath:
                    continue
                filename = filepath.split("/")[-1]
                return f"{filename}:{lineno}"
        return ""

    def _format_args_tuple(self, node: Node) -> str:
        """Format node's tensor args as a tuple for ``_RecordFunctionFast``."""
        parts: list[str] = []

        def collect_nodes(item: Any) -> None:
            if isinstance(item, Node):
                parts.append(item.name)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    collect_nodes(sub_item)

        for arg in node.args:
            collect_nodes(arg)
        for val in node.kwargs.values():
            collect_nodes(val)

        if not parts:
            return "()"
        return f"({', '.join(parts)},)"
