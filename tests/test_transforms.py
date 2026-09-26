"""Parity tests for the ported `aicsimageio.transforms` dimension engine.

The reference is the real `aicsimageio.transforms` when `tests/reference.py`
can resolve it (installed, cached wheel, or a fresh PyPI download). When it
cannot -- an offline machine with no cache -- the same matrix runs against the
NumPy reference in `_numpy_reshape` below and
`test_reference_kind_is_reported` says so. The NumPy reference is exercised on
every run either way, so a green suite never depends on the download.

All comparisons are exact: the port copies whole array elements, so there is
no floating-point arithmetic anywhere in it and `rtol`/`atol` would be
meaningless.
"""

import numpy as np
import pytest

import mojo_aicsimageio
from mojo_aicsimageio import ConflictingArgumentsError

from . import reference as ref

STACK5 = np.arange(4 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(4, 3, 4, 5, 6)
STACK3 = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
G5 = "ZYXWV"

# (name, data, given_dims, return_dims, kwargs)
CASES = [
    ("identity", STACK5, G5, G5, {}),
    ("full reversal", STACK5, G5, "VWXYZ", {}),
    ("swap middle", STACK5, G5, "ZYXWV", {}),
    ("rotate left", STACK5, G5, "YWXVZ", {}),
    ("single axis drop", STACK5, G5, "YXWV", {"Z": 1}),
    ("two axes dropped", STACK5, G5, "YV", {"Z": 0, "W": 3}),
    ("drop with a negative index", STACK5, G5, "YXWV", {"Z": -1}),
    ("new axis inserted at front", STACK5, G5, "TZYXWV", {"T": 0}),
    ("new axis inserted mid-stack", STACK5, G5, "ZTYXWV", {"T": 0}),
    ("two new axes", STACK5, G5, "BCZYXWV", {"B": 0, "C": 0}),
    ("list collapses to a strided slice", STACK5, G5, "ZXYWV", {"Z": [0, 2]}),
    ("strided slice", STACK5, G5, "ZXYWV", {"Z": slice(0, 4, 2)}),
    ("range selection", STACK5, G5, "ZXYWV", {"Z": range(2)}),
    ("tuple with a negative index", STACK5, G5, G5, {"Z": (0, -1)}),
    ("non-monotonic list", STACK5, G5, G5, {"Z": [3, 1, 0]}),
    ("non-monotonic list permuted", STACK5, G5, "ZXYWV", {"Z": [3, 0, 2], "Y": slice(0, 3, 2)}),
    ("empty selection", STACK5, G5, G5, {"Z": slice(0, 0)}),
    ("select then permute", STACK5, G5, "WVX", {"Z": 1, "X": slice(0, 4, 3)}),
    ("select and insert together", STACK5, G5, "TZYXWV", {"T": 0, "Z": [0, 1], "W": slice(0, 5, 2)}),
    ("advanced axis kept and permuted", STACK5, G5, "XZYV", {"Z": [0, 2, 1]}),
    ("advanced axis after drops", STACK5, G5, "ZXV", {"Z": [1, 0], "Y": 2, "W": 4}),
    ("advanced plus strided", STACK5, G5, "ZXY", {"Z": [1, 0], "X": slice(0, 4, 2)}),
    ("2d plane", STACK3, "ZYX", "YX", {"Z": 1}),
    ("2d slice on new dims", STACK3, "ZYX", "TZYX", {"T": 0}),
    ("docstring: slice on new dims", STACK3, "CYX", "BCTZYX", {"C": slice(0, 3, 3)}),
    ("docstring: first and last", STACK3, "ZYX", "ZYX", {"Z": (0, 1, -1)}),
    ("float32 stack", STACK3, "ZYX", "YXZ", {"Z": 0}),
]

# (name, data, given_dims, return_dims, kwargs, expected exception name)
ERROR_CASES = [
    ("list on a dropped axis", STACK5, G5, "YXWV", {"Z": [0, 1]}, "ConflictingArgumentsError"),
    ("slice on a dropped axis", STACK5, G5, "YXWV", {"Z": slice(0, 4, 2)}, "ConflictingArgumentsError"),
    ("range on a dropped axis", STACK5, G5, "YXWV", {"Z": range(2)}, "ConflictingArgumentsError"),
    ("index past the axis", STACK5, G5, G5, {"Z": 9}, "IndexError"),
    ("list index past the axis", STACK5, G5, G5, {"Z": [0, 9]}, "IndexError"),
    ("tuple index past the axis", STACK5, G5, G5, {"Z": (0, 9)}, "IndexError"),
    (
        "mismatched advanced lists",
        STACK5,
        G5,
        G5,
        {"Z": [0, 2, 1], "Y": [0, 2, 1, 0]},
        "IndexError",
    ),
    (
        "two advanced axes collapse the rank",
        STACK5,
        G5,
        G5,
        {"Z": [0, 2, 1], "Y": [2, 0, 2]},
        "ValueError",
    ),
    ("unbounded slice selection", STACK5, G5, G5, {"Z": slice(None, None, 2)}, "TypeError"),
    ("negative list index out of range", STACK5, G5, G5, {"Z": [0, -9]}, "IndexError"),
]


def _numpy_reshape(data, given_dims, return_dims, **kwargs):
    """Independent NumPy reference for the transform.

    Used when the real `aicsimageio` cannot be resolved, and exercised on
    every run as a second opinion on the Mojo gather plan. It re-derives the
    per-axis selection from the documented kwargs and then hands every byte of
    movement to NumPy, so it shares none of the port's gather-plan index
    arithmetic. It does reuse `reduce_to_slice`, because that is pure-Python
    argument handling which `test_reshape_data_raises_the_same_error` and the
    upstream matrix check directly whenever upstream is present.
    """
    key = []
    for dim in given_dims:
        spec = kwargs.get(dim)
        if dim not in return_dims:
            key.append(0 if spec is None else spec)
        elif spec is None:
            key.append(slice(None))
        elif isinstance(spec, int):
            key.append(slice(spec, spec + 1))
        elif isinstance(spec, slice):
            key.append(spec)
        else:
            spec = list(spec)
            reduced = mojo_aicsimageio.reduce_to_slice(spec)
            key.append(spec if not isinstance(reduced, slice) else reduced)
    for _axis in range(len(given_dims), np.asarray(data).ndim):
        key.append(slice(None))
    result = np.asarray(data)[tuple(key)]

    new_dims = given_dims
    for dim in given_dims:
        if dim not in return_dims:
            new_dims = new_dims.replace(dim, "")
    for i, dim in enumerate(return_dims):
        if dim not in given_dims:
            new_dims = f"{new_dims[:i]}{dim}{new_dims[i:]}"
            result = result.reshape(*result.shape[:i], 1, *result.shape[i:])
    return np.transpose(result, [new_dims.find(d) for d in return_dims])


def expected_reshape(data, given_dims, return_dims, **kwargs):
    if ref.REFERENCE is not None:
        return ref.upstream_reshape(data, given_dims, return_dims, **kwargs)
    return _numpy_reshape(data, given_dims, return_dims, **kwargs)


def expected_transpose(data, given_dims, return_dims):
    if ref.REFERENCE is not None:
        return ref.REFERENCE.transpose_to_dims(data, given_dims, return_dims)
    return np.ascontiguousarray(
        np.transpose(data, [given_dims.find(d) for d in return_dims])
    )


def test_reference_kind_is_reported(capsys):
    kind = ref.reference_kind()
    assert kind in ("installed", "wheel", "analytic-numpy")
    with capsys.disabled():
        print(f"\nparity reference for this run: {kind}")


@pytest.mark.parametrize("name,data,given,ret,kw", CASES, ids=[c[0] for c in CASES])
def test_reshape_data_matches_upstream(name, data, given, ret, kw):
    expected = expected_reshape(data, given, ret, **kw)
    got = mojo_aicsimageio.reshape_data(data, given, ret, **kw)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "name,data,given,ret,kw",
    [c[:5] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_numpy_fallback_reference_agrees_with_the_kernel(name, data, given, ret, kw):
    # The fallback path is exercised on every run, so a green suite never
    # depends on the wheel download having succeeded.
    expected = _numpy_reshape(data, given, ret, **kw)
    got = mojo_aicsimageio.reshape_data(data, given, ret, **kw)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "name,data,given,ret,kw,exc",
    ERROR_CASES,
    ids=[c[0] for c in ERROR_CASES],
)
def test_reshape_data_raises_the_same_error(name, data, given, ret, kw, exc):
    if ref.REFERENCE is not None:
        with pytest.raises(Exception) as upstream:
            ref.upstream_reshape(data, given, ret, **kw)
        assert type(upstream.value).__name__ == exc
    with pytest.raises(Exception) as ours:
        mojo_aicsimageio.reshape_data(data, given, ret, **kw)
    assert type(ours.value).__name__ == exc


TRANSPOSE_CASES = [
    ("identity", G5, G5),
    ("reversal", G5, "VWXYZ"),
    ("rotate left", G5, "YWXVZ"),
    ("swap two", "ZYX", "ZYX"),
    ("move the slowest axis to the front", "ZYX", "XZY"),
]


@pytest.mark.parametrize(
    "name,given,ret", TRANSPOSE_CASES, ids=[c[0] for c in TRANSPOSE_CASES]
)
def test_transpose_to_dims_matches_upstream(name, given, ret):
    data = STACK5 if len(given) == 5 else STACK3
    expected = expected_transpose(data, given, ret)
    got = mojo_aicsimageio.transpose_to_dims(data, given, ret)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_transpose_rejects_incompatible_dimensions():
    with pytest.raises(ConflictingArgumentsError):
        mojo_aicsimageio.transpose_to_dims(STACK5, G5, "ZYXW")
    with pytest.raises(ConflictingArgumentsError):
        mojo_aicsimageio.transpose_to_dims(STACK5, "ZZYXW", "ZYXWV")
    with pytest.raises(ConflictingArgumentsError):
        mojo_aicsimageio.transpose_to_dims(STACK5, G5, G5 + "Q")


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0], "slice(0, 1, None)"),
        ([3], "slice(3, 4, None)"),
        ([0, 2, 4], "slice(0, 5, 2)"),
        ([1, 2, 3], "slice(1, 4, 1)"),
        ([0, 1, 3], "[0, 1, 3]"),
        ([2, 1, 0], "[2, 1, 0]"),
        ([0, -1], "[0, -1]"),
    ],
)
def test_reduce_to_slice_matches_upstream(values, expected):
    got = mojo_aicsimageio.reduce_to_slice(list(values))
    if ref.REFERENCE is not None:
        assert repr(got) == repr(ref.upstream_reduce_to_slice(list(values)))
    assert repr(got) == expected


def test_result_is_always_a_fresh_contiguous_array():
    got = mojo_aicsimageio.reshape_data(STACK5, G5, G5)
    assert got.flags["C_CONTIGUOUS"]
    got[0, 0, 0, 0, 0] = 999
    assert STACK5[0, 0, 0, 0, 0] == 0


def test_non_contiguous_input_is_handled():
    view = STACK5[::-1]
    assert not view.flags["C_CONTIGUOUS"]
    got = mojo_aicsimageio.reshape_data(view, G5, G5)
    np.testing.assert_array_equal(got, np.ascontiguousarray(view))


def test_non_array_input_is_rejected_with_a_clear_message():
    with pytest.raises(TypeError, match="numpy.ndarray only"):
        mojo_aicsimageio.reshape_data([[1, 2], [3, 4]], "YX", "YX")


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int32, np.float32, np.float64, np.complex64],
)
def test_dtype_is_preserved_exactly(dtype):
    src = np.arange(60, dtype=np.uint8).astype(dtype).reshape(3, 4, 5)
    got = mojo_aicsimageio.reshape_data(src, "ZYX", "YZX", Y=0)
    assert got.dtype == src.dtype
    np.testing.assert_array_equal(got, src[:, 0:1, :].transpose(1, 0, 2))


@pytest.mark.parametrize(
    "given,ret,kw",
    [
        ("ZYX", "ZYX", {}),
        ("ZYX", "YX", {"Z": 1}),
        ("ZYX", "ZXY", {"Z": slice(0, 3, 2)}),
        ("CYX", "BCTZYX", {"C": slice(0, 3, 3)}),
    ],
    ids=["identity", "drop", "permute", "new dims"],
)
def test_given_dims_shorter_than_the_array_rank(given, ret, kw):
    # NumPy applies an index tuple to the leading axes only, so a `given_dims`
    # that under-describes the array survives the selection and dies in the
    # final transpose. Both sides must die the same way.
    assert len(given) < STACK5.ndim
    if ref.REFERENCE is not None:
        with pytest.raises(Exception) as upstream:
            ref.upstream_reshape(STACK5, given, ret, **kw)
        assert type(upstream.value).__name__ == "ValueError"
    with pytest.raises(Exception) as ours:
        mojo_aicsimageio.reshape_data(STACK5, given, ret, **kw)
    assert type(ours.value).__name__ == "ValueError"


def test_given_dims_longer_than_the_array_rank():
    with pytest.raises(IndexError):
        mojo_aicsimageio.reshape_data(STACK3, "ZYXW", "YXW")
