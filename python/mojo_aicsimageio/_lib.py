"""ctypes bridge to the compiled Mojo gather kernel.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64`; `c_int` truncates and
segfaults on a buffer above 4 GiB. The arity is asserted at load time because a
short argtypes list is zero-filled by ctypes and silently truncates an address.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_LIB_PATH = _ROOT / "dist" / "libmojo-aicsimageio.so"

_GATHER_ARITY = 7

_plan = np.zeros(4096, dtype=np.int64)
_ones_cache = {}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.aics_gather.restype = None
    lib.aics_gather.argtypes = [ctypes.c_int64] * _GATHER_ARITY
    if len(lib.aics_gather.argtypes) != _GATHER_ARITY:
        raise RuntimeError("aics_gather argtypes arity mismatch")
    return lib


lib = _load()


class AxisPlan:
    """One output axis: an extent, and a group of source-offset tables.

    A group holds `len(tables)` back-to-back tables of `size` Int64 entries.
    The common case is one table, which is a plain per-axis selection. A group
    of several is NumPy's advanced-index block, where one output axis reads
    several source axes at the same position of a broadcast index.
    """

    __slots__ = ("size", "tables")

    def __init__(self, size, tables):
        self.size = int(size)
        self.tables = [np.ascontiguousarray(t, dtype=np.int64) for t in tables]

    @property
    def ngroup(self):
        return len(self.tables)


def _contiguous_run(axes):
    """Length of the contiguous source run the fastest output axis reads.

    Returns 0 unless the last axis is a single table whose entries step by
    exactly one. When it does step by one, the whole output is a sequence of
    contiguous runs and the kernel can copy each run sequentially instead of
    chasing an index per element.
    """
    if not axes:
        return 0
    last = axes[-1]
    if last.ngroup != 1 or last.size < 2:
        return 0
    ones = _ones_cache.get(last.size)
    if ones is None:
        ones = np.ones(last.size - 1, dtype=np.int64)
        _ones_cache[last.size] = ones
    if not np.array_equal(np.diff(last.tables[0]), ones):
        return 0
    return last.size


def _run(src, dst, axes, src_base=0):
    """Gather `src` into `dst` following `axes`, using the Mojo kernel."""
    global _plan
    n = len(axes)
    n_out = 1
    for ax in axes:
        n_out *= ax.size
    need = 3 + 5 * n + sum(ax.ngroup * ax.size for ax in axes)
    if _plan.size < need:
        _plan = np.zeros(need + (need >> 1) + 16, dtype=np.int64)
    plan = _plan

    plan[0] = n
    plan[1] = n_out
    plan[2] = src_base
    shape_at = 3
    gstart_at = shape_at + n
    ngroup_at = shape_at + 4 * n
    table_at = shape_at + 5 * n

    cursor = 0
    for d, ax in enumerate(axes):
        plan[shape_at + d] = ax.size
        plan[gstart_at + d] = cursor
        plan[ngroup_at + d] = ax.ngroup
        for t in ax.tables:
            if t.size != ax.size:
                raise ValueError(
                    f"axis {d}: table length {t.size} != extent {ax.size}"
                )
            plan[table_at + cursor: table_at + cursor + ax.size] = t
            cursor += ax.size

    lib.aics_gather(
        ctypes.c_int64(src.ctypes.data),
        ctypes.c_int64(dst.ctypes.data),
        ctypes.c_int64(src.itemsize),
        ctypes.c_int64(n_out),
        ctypes.c_int64(n),
        ctypes.c_int64(plan.ctypes.data),
        ctypes.c_int64(_contiguous_run(axes)),
    )
    return dst
