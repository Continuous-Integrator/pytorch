"""
Stubs for when spmd_types is not installed.

Type constants are copied from spmd_types/types.py.
Everything else is a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PerMeshAxisLocalSpmdType(Enum):
    R = "R"
    I = "I"
    V = "V"
    P = "P"

    def __repr__(self):
        return self.value


@dataclass(frozen=True)
class Shard:
    dim: int

    def __repr__(self):
        return f"S({self.dim})"


R = PerMeshAxisLocalSpmdType.R
I = PerMeshAxisLocalSpmdType.I
V = PerMeshAxisLocalSpmdType.V
P = PerMeshAxisLocalSpmdType.P
S = Shard


class MeshAxis:
    def __init__(self, key: Any = None) -> None:
        self._key = key

    @classmethod
    def of(cls, pg: Any, stride: Any = None) -> MeshAxis:
        return cls(key=id(pg))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MeshAxis) and self._key == other._key


def assert_type(tensor: Any, type: Any = None, **kwargs: Any) -> Any:
    return tensor


def has_local_type(tensor: Any) -> bool:
    return False


def get_local_type(tensor: Any) -> dict:
    return {}


@contextmanager
def typecheck(strict_mode: Any = None, local: Any = None):
    yield


@contextmanager
def set_current_mesh(axes: Any = None):
    yield
