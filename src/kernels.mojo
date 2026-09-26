"""Compiled N-D gather kernels for the `aicsimageio.transforms` dimension engine.

`aicsimageio` has no per-sample arithmetic anywhere: it reads microscopy files,
parses OME metadata and shuffles dimensions with dask/NumPy. The only
array-scale compute in the whole package is what `transforms.reshape_data` and
`transforms.transpose_to_dims` end up doing to pixel data -- an N-dimensional
gather where every output axis either reads a selected set of indices from one
source axis, or reads several source axes at once (NumPy's advanced-index
block, which collapses a set of list selections into a single output axis).

Two loops implement that. When the fastest-varying output axis walks
consecutive source elements, the whole transform is a sequence of contiguous
runs and the copy is a sequential, vectorisable pass. When it does not, an
incremental multi-dimensional odometer drives an index-driven copy, which is
latency-bound anyway. The plan builder picks between them.

The run path is specialised per element width, one loop per width rather than a
width test inside the inner loop: a branch on `elem_size` there is loop
invariant but still keeps the body scalar, and a nested byte loop in the
fallback arm stops the vectoriser outright. Every typed read touches exactly
`elem_size` bytes at the element's own offset, so none of them can run past the
end of the source buffer.

All buffers cross the C ABI as 64-bit addresses and are rebuilt as pointers
inside the body, because `@export` rejects parametric functions and an
inferred pointer origin would make the symbol parametric.
"""

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = Pointer[UInt16, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def i64(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def u8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


# Plan block layout, all Int64. `n` is the output rank.
#
#   0                  n          output rank
#   1                  n_out      number of output elements
#   2                  src_base   constant element offset added to every index
#   3 + d              shape[d]   extent of output axis d
#   3 + n + d          gstart[d]  start of axis d's table group
#   3 + 2n + d         odo[d]     scratch: current index on output axis d
#   3 + 3n + d         noff[d]    scratch: current source offset of axis d
#   3 + 4n + d         ngroup[d]  tables in axis d's group (1, or #advanced)
#   3 + 5n + t         table[t]   index tables, pre-scaled by source stride
#
# An axis group is `ngroup[d]` back-to-back tables of `shape[d]` entries each.
# A plain axis has one table. NumPy's advanced-index block has one table per
# contributing source axis, all advancing in lockstep, which is exactly the
# broadcast-and-sum that block performs. Every table already holds
# `src_index * src_stride`, so advancing an axis is a plain integer add, and a
# freshly inserted length-1 axis degenerates to shape 1 with a zero table.


def group_offset(
    table: I64Ptr, gstart: Int64, shape: Int64, ngroup: Int64, c: Int64
) -> Int64:
    """Sum axis position `c` across every table in the group at `gstart`."""
    var acc = table[gstart + c]
    for j in range(1, Int(ngroup)):
        acc += table[gstart + Int64(j) * shape + c]
    return acc


def odo_advance(
    table: I64Ptr,
    gstart: I64Ptr,
    shape: I64Ptr,
    ngroup: I64Ptr,
    odo: I64Ptr,
    noff: I64Ptr,
    a: Int,
) -> Int64:
    """Advance the odometer over axes `a..0` and return the source offset moved.

    Called once per run, not once per element, so it does not need to inline.
    """
    var moved = Int64(0)
    var b = a
    while b >= 0:
        var c = odo[b] + Int64(1)
        var old = noff[b]
        if c < shape[b]:
            odo[b] = c
            noff[b] = group_offset(
                table, gstart[b], shape[b], ngroup[b], c
            )
            moved += noff[b] - old
            return moved
        odo[b] = Int64(0)
        noff[b] = group_offset(
            table, gstart[b], shape[b], ngroup[b], Int64(0)
        )
        moved += noff[b] - old
        b -= 1
    return moved


def run_copy_u8(
    sp: U8Ptr, dp: U8Ptr, n_runs: Int, inner: Int, run_base: Int,
    table: I64Ptr, gstart: I64Ptr, shape: I64Ptr, ngroup: I64Ptr,
    odo: I64Ptr, noff: I64Ptr, n: Int,
):
    var doff = 0
    var rb = run_base
    for _ in range(n_runs):
        for j in range(inner):
            dp[unsafe_offset=doff + j] = sp[unsafe_offset=rb + j]
        doff += inner
        rb += Int(odo_advance(table, gstart, shape, ngroup, odo, noff, n - 2))


def run_copy_u16(
    sp: U16Ptr, dp: U16Ptr, n_runs: Int, inner: Int, run_base: Int,
    table: I64Ptr, gstart: I64Ptr, shape: I64Ptr, ngroup: I64Ptr,
    odo: I64Ptr, noff: I64Ptr, n: Int,
):
    var doff = 0
    var rb = run_base
    for _ in range(n_runs):
        for j in range(inner):
            dp[unsafe_offset=doff + j] = sp[unsafe_offset=rb + j]
        doff += inner
        rb += Int(odo_advance(table, gstart, shape, ngroup, odo, noff, n - 2))


def run_copy_u32(
    sp: U32Ptr, dp: U32Ptr, n_runs: Int, inner: Int, run_base: Int,
    table: I64Ptr, gstart: I64Ptr, shape: I64Ptr, ngroup: I64Ptr,
    odo: I64Ptr, noff: I64Ptr, n: Int,
):
    var doff = 0
    var rb = run_base
    for _ in range(n_runs):
        for j in range(inner):
            dp[unsafe_offset=doff + j] = sp[unsafe_offset=rb + j]
        doff += inner
        rb += Int(odo_advance(table, gstart, shape, ngroup, odo, noff, n - 2))


def run_copy_u64(
    sp: U64Ptr, dp: U64Ptr, n_runs: Int, inner: Int, run_base: Int,
    table: I64Ptr, gstart: I64Ptr, shape: I64Ptr, ngroup: I64Ptr,
    odo: I64Ptr, noff: I64Ptr, n: Int,
):
    var doff = 0
    var rb = run_base
    for _ in range(n_runs):
        for j in range(inner):
            dp[unsafe_offset=doff + j] = sp[unsafe_offset=rb + j]
        doff += inner
        rb += Int(odo_advance(table, gstart, shape, ngroup, odo, noff, n - 2))


@export("aics_gather")
def aics_gather(
    src_addr: Int, dst_addr: Int, elem_size: Int, n_out: Int, n: Int,
    plan_addr: Int, inner: Int
) abi("C"):
    """Copy `n_out` elements of `elem_size` bytes from `src` to `dst`.

    The source index of every output element comes from the plan at
    `plan_addr`; see the block layout above. Elements are copied whole and
    uninterpreted, so the result is bit-exact for every dtype.

    `inner` is the length of the run of consecutive source elements that the
    fastest-varying output axis reads, or 0 when it does not.
    """
    if n <= 0 or n_out <= 0:
        return
    var plan = i64(plan_addr)
    var shape = plan + 3
    var gstart = plan + 3 + n
    var odo = plan + 3 + 2 * n
    var noff = plan + 3 + 3 * n
    var ngroup = plan + 3 + 4 * n
    var table = plan + 3 + 5 * n

    for d in range(n):
        odo[d] = Int64(0)
        noff[d] = group_offset(
            table, gstart[d], shape[d], ngroup[d], Int64(0)
        )

    var total = 0
    for d in range(n):
        total += Int(noff[d])
    var base = Int(plan[unsafe_offset=2])

    if inner > 0:
        # The fastest axis walks consecutive source elements, so the transform
        # is a sequence of contiguous runs: the last axis always sits at
        # position 0 and contributes table[gstart[n-1]] for every run, and
        # only the outer odometer moves.
        var run_base = base + total - Int(noff[n - 1]) + Int(
            table[gstart[n - 1]]
        )
        var n_runs = n_out // inner
        if elem_size == 1:
            run_copy_u8(
                U8Ptr(unsafe_from_address=src_addr),
                U8Ptr(unsafe_from_address=dst_addr),
                n_runs, inner, run_base, table, gstart, shape, ngroup,
                odo, noff, n,
            )
        elif elem_size == 2:
            run_copy_u16(
                U16Ptr(unsafe_from_address=src_addr),
                U16Ptr(unsafe_from_address=dst_addr),
                n_runs, inner, run_base, table, gstart, shape, ngroup,
                odo, noff, n,
            )
        elif elem_size == 4:
            run_copy_u32(
                U32Ptr(unsafe_from_address=src_addr),
                U32Ptr(unsafe_from_address=dst_addr),
                n_runs, inner, run_base, table, gstart, shape, ngroup,
                odo, noff, n,
            )
        elif elem_size == 8:
            run_copy_u64(
                U64Ptr(unsafe_from_address=src_addr),
                U64Ptr(unsafe_from_address=dst_addr),
                n_runs, inner, run_base, table, gstart, shape, ngroup,
                odo, noff, n,
            )
        else:
            # Odd widths NumPy allows: 3, 6, 16 bytes and so on. Scaling the
            # run into bytes lets the u8 loop carry them.
            run_copy_u8(
                U8Ptr(unsafe_from_address=src_addr),
                U8Ptr(unsafe_from_address=dst_addr),
                n_runs, inner * elem_size, run_base * elem_size, table,
                gstart, shape, ngroup, odo, noff, n,
            )
        return

    # Index-driven gather. O(1) amortised per element: only the
    # fastest-moving axis normally takes the odometer branch. Each axis
    # contributes an independent part of the flat source offset, so the
    # running total is carried forward by the per-axis delta.
    var sp = u8(src_addr)
    var dp = u8(dst_addr)
    var sp16 = U16Ptr(unsafe_from_address=src_addr)
    var dp16 = U16Ptr(unsafe_from_address=dst_addr)
    var sp32 = U32Ptr(unsafe_from_address=src_addr)
    var dp32 = U32Ptr(unsafe_from_address=dst_addr)
    var sp64 = U64Ptr(unsafe_from_address=src_addr)
    var dp64 = U64Ptr(unsafe_from_address=dst_addr)
    var soff = base + total
    var doff = 0
    for _ in range(n_out):
        if elem_size == 1:
            dp[unsafe_offset=doff] = sp[unsafe_offset=soff]
        elif elem_size == 2:
            dp16[unsafe_offset=doff] = sp16[unsafe_offset=soff]
        elif elem_size == 4:
            dp32[unsafe_offset=doff] = sp32[unsafe_offset=soff]
        elif elem_size == 8:
            dp64[unsafe_offset=doff] = sp64[unsafe_offset=soff]
        else:
            for b in range(elem_size):
                dp[unsafe_offset=doff * elem_size + b] = sp[
                    unsafe_offset=soff * elem_size + b
                ]
        doff += 1
        var a = n - 1
        while a >= 0:
            var c = odo[a] + Int64(1)
            var old = noff[a]
            if c < shape[a]:
                odo[a] = c
                noff[a] = group_offset(
                    table, gstart[a], shape[a], ngroup[a], c
                )
                soff += Int(noff[a] - old)
                break
            odo[a] = Int64(0)
            noff[a] = group_offset(
                table, gstart[a], shape[a], ngroup[a], Int64(0)
            )
            soff += Int(noff[a] - old)
            a -= 1
