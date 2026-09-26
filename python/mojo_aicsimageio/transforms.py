"""Mojo-backed port of `aicsimageio.transforms`.

Upstream `aicsimageio` has no per-sample arithmetic at all: it reads
microscopy files, parses OME metadata and moves dimensions around with
dask/NumPy. The only array-scale compute in the package is the dimension
engine in `transforms.py`, and that is what this module reimplements on top of
the compiled N-D gather in `src/kernels.mojo`.

The Python-side argument handling below is a faithful transcription of upstream:
the same conflict checks, the same `reduce_to_slice` conversion, the same
`IndexError` / `TypeError` on malformed selections, and the same `new_dims`
string bookkeeping. Only the pixel movement differs: instead of
`data[tuple(dim_specs)]` followed by `.transpose(...)`, the whole transform
compiles into a single gather plan and goes through one Mojo call.

Two deliberate, documented differences from upstream:

* upstream may return a *view* into the input when every selection is a basic
  slice; this port always returns a fresh C-contiguous array, because the Mojo
  kernel writes into a caller-owned destination.
* upstream accepts dask and xarray arrays. This port accepts NumPy arrays only
  and raises `TypeError` otherwise.
"""

from collections import Counter

import numpy as np

from ._lib import AxisPlan, _run
from .exceptions import ConflictingArgumentsError


def reduce_to_slice(L):
    """Collapse an evenly spaced, ascending index list into a slice.

    Transcription of `aicsimageio.transforms.reduce_to_slice`, including the
    quirk that the original list object is handed back untouched when the
    deltas are not uniform and positive.
    """
    if len(L) == 1:
        return slice(L[0], L[0] + 1)
    steps = [(L[i + 1] - L[i]) for i in range(len(L) - 1)]
    if steps[0] > 0 and steps.count(steps[0]) == len(steps):
        return slice(min(L), max(L) + 1, steps[0])
    return L


def _c_strides(shape):
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    strides.reverse()
    return strides


def _slice_indices(spec, n):
    start, stop, step = spec.indices(n)
    return list(range(start, stop, step))


def _list_indices(values, n, axis):
    out = []
    for v in values:
        j = v + n if v < 0 else v
        if j < 0 or j >= n:
            raise IndexError(
                f"index {v} is out of bounds for axis {axis} with size {n}"
            )
        out.append(j)
    return out


def _selected_axes(shape, specs):
    """Model `data[tuple(dim_specs)]` as a list of gather axes.

    `specs` is one entry per source axis: `None` for a dropped axis, otherwise
    `(indices, advanced)`. Returns `(size, [tables])` per surviving axis, in the
    axis order the selection actually produces, each table already scaled by
    its source stride.

    NumPy collapses two or more list selections into a single leading output
    axis that broadcasts across all of them, and leaves a lone list selection
    in place. That rule is reproduced here as an axis *group*: several stride-
    scaled tables of equal length, summed. Upstream never gets to use it --
    with two or more advanced axes its own `new_dims` bookkeeping cannot
    describe the collapsed rank, so it raises `ValueError` from
    `ndarray.transpose` -- but the kernel supports it and it is what makes the
    port correct rather than accidentally correct.
    """
    strides = _c_strides(shape)
    kept = [(a, spec) for a, spec in enumerate(specs) if spec is not None]
    adv = [(a, spec) for a, spec in kept if spec[1]]

    if len(adv) <= 1:
        return [
            (len(ix), [np.asarray(ix, dtype=np.int64) * strides[a]])
            for a, (ix, _adv) in kept
        ]

    width = max(len(ix) for _a, (ix, _adv) in adv)
    tables = []
    for a, (ix, _adv) in adv:
        if len(ix) not in (1, width):
            raise IndexError(
                "shape mismatch: indexing arrays could not be broadcast "
                f"together with shapes ({len(ix)},) ({width},)"
            )
        grown = ix if len(ix) == width else [ix[p] for p in range(width)]
        tables.append(np.asarray(grown, dtype=np.int64) * strides[a])
    out = [(width, tables)]
    for a, (ix, adv_flag) in kept:
        if not adv_flag:
            out.append(
                (len(ix), [np.asarray(ix, dtype=np.int64) * strides[a]])
            )
    return out


def _run_plan(data, axes, src_base=0):
    src = np.ascontiguousarray(data)
    shape = [s for s, _t in axes]
    n_out = int(np.prod(shape, dtype=np.int64)) if shape else 1
    dst = np.empty(n_out, dtype=src.dtype)
    if n_out == 0:
        return dst.reshape(shape)
    _run(src, dst, [AxisPlan(s, t) for s, t in axes], src_base=src_base)
    return dst.reshape(shape)


def _full_transpose_axes(shape, transposer):
    # NumPy's own `ndarray.transpose` refuses a permutation that does not
    # cover every axis, and upstream lets that error through; so do we.
    if len(transposer) != len(shape):
        raise ValueError("axes don't match array")
    strides = _c_strides(shape)
    axes = []
    for t in transposer:
        if t < 0 or t >= len(shape):
            raise ValueError("axes don't match array")
        n = shape[t]
        axes.append((n, [np.arange(n, dtype=np.int64) * strides[t]]))
    return axes


def transpose_to_dims(data, given_dims, return_dims):
    """Reshape the dimension order of `data` from `given_dims` to `return_dims`.

    Port of `aicsimageio.transforms.transpose_to_dims`; the permutation is
    executed by the Mojo gather rather than by `ndarray.transpose`.
    """
    if (
        Counter(given_dims) != Counter(return_dims)
        or max(Counter(given_dims).values()) > 1
    ):
        raise ConflictingArgumentsError(
            f"given_dims={given_dims} and return_dims={return_dims} "
            "are incompatible."
        )
    data = np.asarray(data)
    transposer = [given_dims.find(dim) for dim in return_dims]
    return _run_plan(data, _full_transpose_axes(data.shape, transposer))


def reshape_data(data, given_dims, return_dims, **kwargs):
    """Reshape `data` into `return_dims`, padding and pruning as needed.

    Port of `aicsimageio.transforms.reshape_data`. Per-axis selection,
    dimension padding and dimension reordering all compile into one Mojo
    gather plan, so the whole transform is a single pass over the output.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "mojo_aicsimageio.reshape_data accepts numpy.ndarray only; "
            f"got {type(data).__name__}"
        )
    for dim in given_dims:
        if (
            isinstance(kwargs.get(dim), (list, tuple, range, slice))
            and dim not in return_dims
        ):
            raise ConflictingArgumentsError(
                "When selecting a multiple dimension indices, the specified "
                f"dimension must be provided in return_dims. "
                f"return_dims={return_dims}, dimension {dim} = {kwargs.get(dim)}"
            )

    strides = _c_strides(data.shape)
    new_dims = given_dims
    specs = []
    src_base = 0
    for dim in given_dims:
        dim_index = given_dims.index(dim)
        # Upstream leaves this unbound for a spec that is neither an int, a
        # list/tuple/range nor a slice, and dies with UnboundLocalError. The
        # zero default keeps that path unreachable rather than papering over
        # a check that would then be skipped silently.
        check_selection_max = 0
        advanced = False
        if dim in return_dims:
            if dim in kwargs:
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec
                if isinstance(dim_spec, int):
                    dim_spec = slice(dim_spec, dim_spec + 1)
                if isinstance(dim_spec, (tuple, range)):
                    dim_spec = list(dim_spec)
                if isinstance(dim_spec, list):
                    check_selection_max = max(
                        [abs(min(dim_spec)), max(dim_spec)]
                    )
                    dim_spec = reduce_to_slice(dim_spec)
                    advanced = not isinstance(dim_spec, slice)
                if isinstance(dim_spec, slice):
                    check_selection_max = max(
                        [abs(dim_spec.stop), abs(dim_spec.start)]
                    )
            else:
                dim_spec = slice(None, None, None)
                display_dim_spec = dim_spec
                check_selection_max = 0
        else:
            if dim in kwargs:
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec
                check_selection_max = dim_spec
            else:
                dim_spec = 0
                display_dim_spec = dim_spec
                check_selection_max = 0
            new_dims = new_dims.replace(dim, "")

        if check_selection_max > data.shape[dim_index]:
            raise IndexError(
                f"Dimension specified with {dim}={display_dim_spec} "
                f"but Dimension shape is {data.shape[dim_index]}."
            )

        n = data.shape[dim_index]
        if dim not in return_dims:
            # An integer selection drops the axis but still contributes a
            # constant to every source index, which rides on src_base.
            src_base += _list_indices([dim_spec], n, dim_index)[0] * strides[dim_index]
            specs.append(None)
        elif advanced:
            specs.append((_list_indices(dim_spec, n, dim_index), True))
        else:
            specs.append((_slice_indices(dim_spec, n), False))

    # NumPy applies an index tuple to the leading axes only and leaves the rest
    # untouched; upstream inherits that, so a `given_dims` shorter than the
    # array rank survives here and fails later in the transpose instead.
    for axis in range(len(given_dims), data.ndim):
        specs.append((list(range(data.shape[axis])), False))

    sel = _selected_axes(data.shape, specs)

    for i, dim in enumerate(return_dims):
        if dim not in given_dims:
            new_dims = f"{new_dims[:i]}{dim}{new_dims[i:]}"
            sel.insert(i, (1, [np.zeros(1, dtype=np.int64)]))

    transposer = [new_dims.find(dim) for dim in return_dims]
    if len(transposer) != len(sel):
        raise ValueError(
            "axes don't match array: two or more list selections collapse "
            "into one axis, which the dimension string cannot describe"
        )
    return _run_plan(data, [sel[t] for t in transposer], src_base=src_base)
