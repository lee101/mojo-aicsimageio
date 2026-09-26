"""Direct tests of the compiled N-D gather.

Every comparison here is a whole-element byte copy, so `rtol`/`atol` are
meaningless: the kernel moves memory, it does not compute on it, and exact
equality is the correct contract.
"""

import ctypes
import gc
import mmap

import numpy as np
import pytest

import mojo_aicsimageio
from mojo_aicsimageio._lib import AxisPlan, _run, lib


def test_exported_symbol_is_present():
    assert "aics_gather" in mojo_aicsimageio.kernels()


def test_argtypes_arity_is_declared():
    # A ctypes argtypes list one entry short is zero-filled and silently
    # truncates a 64-bit address. Assert the full arity is declared.
    assert len(lib.aics_gather.argtypes) == 7
    assert all(t is ctypes.c_int64 for t in lib.aics_gather.argtypes)


def _identity_axes(n):
    return [AxisPlan(n, [np.arange(n, dtype=np.int64)])]


@pytest.mark.parametrize("itemsize", [1, 2, 4, 8, 3, 6, 16])
def test_every_dtype_width_is_copied_verbatim(itemsize):
    # 1/2/4/8 bytes take the typed paths, 3/6/16 the generic byte path. A void
    # dtype of the right width gives the kernel an `itemsize` it can branch on
    # without the test having to invent a typed array for each width.
    raw = np.resize(np.arange(1, 101, dtype=np.uint8), itemsize * 100)
    src = np.frombuffer(raw.tobytes(), dtype=np.dtype(f"V{itemsize}"))
    out = np.empty(100, dtype=src.dtype)
    _run(src, out, _identity_axes(100))
    assert out.tobytes() == src.tobytes()


@pytest.mark.parametrize("n", [1003, 1005, 1007, 1009, 1011, 1, 2, 7])
def test_tail_is_not_dropped_at_awkward_lengths(n):
    # Sizes that straddle a SIMD width are where a vectorised kernel loses its
    # tail. The odometer here is scalar, so the check is that the last element
    # is still written and the run does not stop one short.
    src = np.arange(n, dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    _run(src, out, _identity_axes(n))
    assert np.array_equal(out, src)


def test_transpose_of_a_prime_shaped_square():
    n = 37
    src = np.arange(n * n, dtype=np.int64)
    out = np.empty(n * n, dtype=np.int64)
    axes = [
        AxisPlan(n, [np.arange(n, dtype=np.int64) * 1]),
        AxisPlan(n, [np.arange(n, dtype=np.int64) * n]),
    ]
    _run(src, out, axes)
    assert np.array_equal(out.reshape(n, n), src.reshape(n, n).T)


def test_group_of_three_tables_is_the_broadcast_advanced_block():
    # NumPy gives `a[idx, idx]` a single block axis where position p takes
    # element p of every index array at once. That is a sum of per-axis
    # tables, which is exactly what an axis group expresses.
    src = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    pick = np.asarray([1, 0])
    tables = [pick * stride for stride in (12, 4, 1)]
    out = np.empty(2, dtype=np.int64)
    _run(src.reshape(-1), out, [AxisPlan(2, tables)])
    assert np.array_equal(out, src[pick, pick, pick])


def test_group_with_a_broadcast_axis_of_length_one_is_pre_expanded():
    # `src[0, [0, 1]]` broadcasts the length-1 index against the length-2 one.
    # The plan layer pre-expands it; the kernel only ever sees equal lengths.
    src = np.arange(12, dtype=np.int64).reshape(3, 4)
    grown = np.asarray([0, 0], dtype=np.int64)
    tables = [grown * 4, np.asarray([0, 1], dtype=np.int64) * 1]
    out = np.empty(2, dtype=np.int64)
    _run(src.reshape(-1), out, [AxisPlan(2, tables)])
    assert np.array_equal(out, src[np.asarray([0]), np.asarray([0, 1])].reshape(2))


def test_group_table_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="table length"):
        _run(
            np.arange(8, dtype=np.int64),
            np.empty(2, dtype=np.int64),
            [AxisPlan(2, [np.zeros(2, dtype=np.int64), np.zeros(1, dtype=np.int64)])],
        )


def test_src_base_offsets_every_index():
    src = np.arange(10, dtype=np.int64)
    out = np.empty(3, dtype=np.int64)
    _run(src, out, _identity_axes(3), src_base=5)
    assert np.array_equal(out, src[5:8])


def test_zero_length_axis_writes_nothing():
    src = np.arange(10, dtype=np.int64)
    out = np.empty(0, dtype=np.int64)
    _run(src, out, [AxisPlan(0, [np.zeros(0, dtype=np.int64)])])
    assert out.size == 0


def test_buffer_above_four_gib_survives_the_call():
    # The failure this guards: with c_int argtypes the address is truncated
    # and the kernel reads near zero. Small arrays never notice.
    values = np.arange(4096, dtype=np.uint16)
    region = mmap.mmap(-1, values.nbytes)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    assert addr >= 2**32, (
        f"anonymous mmap landed at {addr:#x}, below 4 GiB; this test would "
        "not exercise a truncated address"
    )
    try:
        buf = np.frombuffer(region, dtype=np.uint16, count=values.size)
        buf[:] = values
        out = np.empty(4096, dtype=np.uint16)
        _run(buf, out, _identity_axes(4096))
        assert np.array_equal(out, values)
        del buf
    finally:
        gc.collect()
        region.close()


def test_reversed_index_tables_walk_the_odometer_correctly():
    # Descending index tables: an odometer that confuses its current position
    # with its initial value produces garbage on the wrap.
    n, m = 5, 7
    src = np.arange(n * m, dtype=np.int64)
    out = np.empty(n * m, dtype=np.int64)
    axes = [
        AxisPlan(n, [np.arange(n - 1, -1, -1, dtype=np.int64) * m]),
        AxisPlan(m, [np.arange(m - 1, -1, -1, dtype=np.int64)]),
    ]
    _run(src, out, axes)
    assert np.array_equal(out.reshape(n, m), src.reshape(n, m)[::-1, ::-1])


def test_extent_one_axis_contributes_nothing():
    src = np.arange(5, dtype=np.int64)
    out = np.empty(1, dtype=np.int64)
    _run(src, out, [AxisPlan(1, [np.zeros(1, dtype=np.int64)])], src_base=3)
    assert np.array_equal(out, src[3:4])


def test_non_monotonic_table_exercises_every_branch():
    # A table that jumps around proves the odometer tracks the table, not the
    # axis position.
    table = np.asarray([4, 0, 3, 1, 2, 1, 0], dtype=np.int64)
    src = np.arange(5, dtype=np.int64)
    out = np.empty(7, dtype=np.int64)
    _run(src, out, [AxisPlan(7, [table])])
    assert np.array_equal(out, src[table])
