"""
Dynamic shape specification types for torch.compile and torch.export.

Provides IntSpec for fine-grained control over whether an integer dimension
is dynamic (backed), static, or unbacked.

Usage::

    # Direct construction
    IntSpec("batch", type=IntSpecType.BACKED, min=1, max=64)
    IntSpec("heads", type=IntSpecType.STATIC, value=8)

    # Fluent API
    IntSpec("batch").backed(min=1, max=64)
    IntSpec().static(10)
    IntSpec("seq_len").unbacked(min=1, max=2048)
"""

import enum
from typing import Any


__all__ = ["IntSpecType", "IntSpec"]


class IntSpecType(enum.Enum):
    """How an integer dimension should be treated during compilation.

    STATIC  -- Treat as a compile-time constant. Recompiles if the value changes.
    BACKED  -- Symbolic with a backing hint. Specialization (including full) is
               permitted; no constraints are enforced at runtime.
    UNBACKED -- Fully symbolic with no backing value. Runtime assertions enforce
                min/max bounds. Cannot be specialized.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Constructed directly or via the fluent API::

        IntSpec("batch", type=IntSpecType.BACKED, min=1, max=64)
        IntSpec("batch").backed(min=1, max=64)
        IntSpec().static(10)

    Parameter validity by type:

    ============  =========  =========  ========  ================  ===========
    Parameter     STATIC     BACKED     UNBACKED
    ============  =========  =========  ========  ================  ===========
    value         yes        NO         NO
    min / max     NO         yes        yes
    backed_hint   NO         yes        NO
    optimization_hint  NO    NO         yes
    ============  =========  =========  ========  ================  ===========
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        type: IntSpecType | None = None,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        optimization_hint: int | None = None,
        backed_hint: int | None = None,
    ) -> None:
        self.name = name
        self._type = type
        self._min = min
        self._max = max
        self._value = value
        self._optimization_hint = optimization_hint
        self._backed_hint = backed_hint
        if type is not None:
            self._validate()

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        if self._type is None:
            return
        if self._type == IntSpecType.STATIC:
            if self._min is not None or self._max is not None:
                raise ValueError(
                    "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC"
                )
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
            if self._backed_hint is not None:
                raise ValueError("backed_hint is only valid for BACKED IntSpec")
        elif self._type == IntSpecType.BACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
            if (
                self._min is not None
                and self._max is not None
                and self._min > self._max
            ):
                raise ValueError(
                    f"min must be <= max, got min={self._min}, max={self._max}"
                )
        elif self._type == IntSpecType.UNBACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._backed_hint is not None:
                raise ValueError("backed_hint is only valid for BACKED IntSpec")
            if (
                self._min is not None
                and self._max is not None
                and self._min > self._max
            ):
                raise ValueError(
                    f"min must be <= max, got min={self._min}, max={self._max}"
                )

    # -- read-only properties ----------------------------------------------

    @property
    def type(self) -> IntSpecType | None:
        return self._type

    @property
    def min(self) -> int | None:
        return self._min

    @property
    def max(self) -> int | None:
        return self._max

    @property
    def value(self) -> int | None:
        return self._value

    @property
    def optimization_hint(self) -> int | None:
        return self._optimization_hint

    @property
    def backed_hint(self) -> int | None:
        return self._backed_hint

    # -- fluent API --------------------------------------------------------

    def static(self, value: int | None = None) -> "IntSpec":
        """Configure as STATIC.  *value* pins the concrete size; if ``None`` the
        value is taken from the example input at compile time."""
        self._type = IntSpecType.STATIC
        self._value = value
        self._min = None
        self._max = None
        self._optimization_hint = None
        self._backed_hint = None
        return self

    def backed(
        self,
        *,
        min: int | None = None,
        max: int | None = None,
        hint: int | None = None,
    ) -> "IntSpec":
        """Configure as BACKED (symbolic with backing hint).
        Specialization is permitted; *min*/*max* are assumptions, not hard
        constraints."""
        self._type = IntSpecType.BACKED
        self._min = min
        self._max = max
        self._backed_hint = hint
        self._value = None
        self._optimization_hint = None
        self._validate()
        return self

    def unbacked(
        self,
        *,
        min: int | None = None,
        max: int | None = None,
        hint: int | None = None,
    ) -> "IntSpec":
        """Configure as UNBACKED (fully symbolic).
        *min*/*max* become runtime assertions.  *hint* is an optimization hint
        only (e.g. for inductor autotuning)."""
        self._type = IntSpecType.UNBACKED
        self._min = min
        self._max = max
        self._optimization_hint = hint
        self._value = None
        self._backed_hint = None
        self._validate()
        return self

    # -- lowering to existing Dim infrastructure ---------------------------

    def _to_dim(self) -> Any:
        """Convert to the existing ``Dim`` / ``_DimHint`` representation
        understood by ``_process_dynamic_shapes``."""
        from torch.export.dynamic_shapes import Dim

        if self._type == IntSpecType.STATIC:
            if self._value is not None:
                return self._value  # _process_dynamic_shapes treats int as _StaticDim
            return Dim.STATIC
        elif self._type == IntSpecType.BACKED:
            # Backed ≈ Dim.AUTO (allows specialization).
            # If min/max are given, create a named Dim with bounds.
            if self._min is not None or self._max is not None:
                dim_name = self.name or "_"
                kwargs: dict[str, int] = {}
                if self._min is not None:
                    kwargs["min"] = self._min
                if self._max is not None:
                    kwargs["max"] = self._max
                return Dim(dim_name, **kwargs)
            return Dim.AUTO
        elif self._type == IntSpecType.UNBACKED:
            return Dim.DYNAMIC
        return None

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self._type is not None:
            parts.append(f"type={self._type.value}")
        if self._value is not None:
            parts.append(f"value={self._value}")
        if self._min is not None:
            parts.append(f"min={self._min}")
        if self._max is not None:
            parts.append(f"max={self._max}")
        if self._optimization_hint is not None:
            parts.append(f"optimization_hint={self._optimization_hint}")
        if self._backed_hint is not None:
            parts.append(f"backed_hint={self._backed_hint}")
        return f"IntSpec({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntSpec):
            return NotImplemented
        return (
            self.name == other.name
            and self._type == other._type
            and self._min == other._min
            and self._max == other._max
            and self._value == other._value
            and self._optimization_hint == other._optimization_hint
            and self._backed_hint == other._backed_hint
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self._type,
                self._min,
                self._max,
                self._value,
                self._optimization_hint,
                self._backed_hint,
            )
        )


def _apply_intspec_to_tensor(tensor: Any, shape_spec: Any) -> None:
    """Apply per-dimension IntSpec entries to a tensor via mark_static/mark_dynamic/mark_unbacked."""
    from torch._dynamo.decorators import mark_static, mark_unbacked, maybe_mark_dynamic

    if isinstance(shape_spec, dict):
        items = shape_spec.items()
    elif isinstance(shape_spec, (list, tuple)):
        items = enumerate(shape_spec)
    else:
        return

    for idx, spec in items:
        if spec is None:
            continue
        if not isinstance(spec, IntSpec):
            raise TypeError(
                f"Expected IntSpec or None in dynamic_shapes, got {type(spec).__name__}"
            )
        if spec.type is None:
            raise ValueError(f"IntSpec type must be set for dim {idx}")
        if spec.type == IntSpecType.STATIC:
            mark_static(tensor, idx)
        elif spec.type == IntSpecType.BACKED:
            maybe_mark_dynamic(tensor, idx)
        elif spec.type == IntSpecType.UNBACKED:
            mark_unbacked(tensor, idx)


def _apply_dynamic_shapes(
    compiled: Any, original: Any, dynamic_shapes: dict[str, Any]
) -> Any:
    """Wrap a compiled callable to apply dynamic_shapes IntSpec on each call.

    The wrapper is decorated with ``torch._dynamo.disable`` so that dynamo
    does not attempt to trace the tensor-marking logic.  The inner
    ``compiled()`` call re-enters dynamo normally.
    """
    import functools
    import inspect

    import torch
    import torch._dynamo

    sig = inspect.signature(
        original.forward if isinstance(original, torch.nn.Module) else original
    )

    # torch._dynamo.disable prevents dynamo from tracing the wrapper itself.
    # When the wrapper calls compiled(), the compiled function re-enables
    # dynamo's frame evaluation for actual tracing.
    @torch._dynamo.disable
    @functools.wraps(
        compiled if not isinstance(compiled, torch.nn.Module) else compiled.forward
    )
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_shapes.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_intspec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    if isinstance(compiled, torch.nn.Module):
        compiled.forward = wrapper  # type: ignore[method-assign]
        return compiled
    return wrapper
