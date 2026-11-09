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
    # Degenerate parity additions (empty / one-element / duplicate /
    # mismatched-shape). These entries also run through the NumPy fallback
    # test below, so they must be expressible by both references; cases where
    # upstream and the analytic reference disagree (duplicate dims, 0-d,
    # non-integer kwargs) are covered by the dedicated tests at the bottom of
    # this file, not here.
    ("empty: zero-extent identity", np.zeros((0, 3, 4), dtype=np.uint16), "ZYX", "ZYX", {}),
    ("empty: zero-extent permute", np.zeros((0, 3, 4), dtype=np.uint16), "ZYX", "YXZ", {}),
    ("empty: all zero extents", np.zeros((0, 0, 0), dtype=np.uint16), "ZYX", "YXZ", {}),
    ("empty: zero-extent strided slice", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": slice(0, 0)}),
    ("one element: 1d identity", np.array([42], dtype=np.uint16), "Z", "Z", {}),
    ("one element: 2d identity", np.array([42], dtype=np.uint16).reshape(1, 1), "ZY", "YZ", {}),
    ("one element: 5d reversal", np.array([99], dtype=np.uint16).reshape(1, 1, 1, 1, 1), "ZYXWV", "VWXYZ", {}),
    ("one element: 5d drop", np.array([99], dtype=np.uint16).reshape(1, 1, 1, 1, 1), "ZYXWV", "YXWV", {"Z": 0}),
    ("duplicates: repeated index preserved", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "YZ", {"Z": [1, 1, 1]}),
    ("duplicates: repeated index permuted", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "YZ", {"Z": [2, 2, 0]}),
    ("duplicates: length-1 advanced list", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "YZ", {"Z": [1]}),
    ("bool kwarg selects like int", np.arange(6, dtype=np.uint16).reshape(2, 3), "ZY", "YZ", {"Z": True}),
    ("range with step", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": range(0, 3, 2)}),
    ("range descending", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": range(2, -1, -1)}),
    ("slice with negative bounds", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": slice(-2, -1)}),
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
    ("empty: drop on zero-extent axis", np.zeros((0, 3, 4), dtype=np.uint16), "ZYX", "YX", {}, "IndexError"),
    ("empty list selection", STACK5, G5, G5, {"Z": []}, "ValueError"),
    ("slice step zero", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": slice(0, 3, 0)}, "ValueError"),
    ("slice with None bound", STACK5, G5, G5, {"Z": slice(None, 2)}, "TypeError"),
    ("int past the axis after drop", STACK5, G5, "YXWV", {"Z": 9}, "IndexError"),
    ("slice with huge stop", np.arange(12, dtype=np.uint16).reshape(3, 4), "ZY", "ZY", {"Z": slice(0, 100)}, "IndexError"),
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


# ---------------------------------------------------------------------------
# Degenerate-input parity: empty, one element, duplicates, NaN/inf,
# mismatched shapes. Value cases use the written tolerance rtol=1e-6,
# atol=1e-6: the kernel copies whole elements verbatim, so an exact match is
# expected and the tolerance only pins the contract (no arithmetic is
# performed whose error would need a looser bound). Error cases compare the
# exception class against upstream directly, with the upstream file+line
# cited, because the analytic NumPy reference cannot express upstream's
# dimension-string bugs.
# ---------------------------------------------------------------------------
def test_nan_inf_data_moves_bit_exact_with_written_tolerance():
    src = np.array([np.nan, np.inf, -np.inf, 3.5], dtype=np.float32).reshape(4, 1)
    got = mojo_aicsimageio.reshape_data(src, "ZY", "YZ")
    assert got.shape == (1, 4)
    np.testing.assert_allclose(got, src.T, rtol=1e-6, atol=1e-6, equal_nan=True)
    if ref.REFERENCE is not None:
        expected = ref.upstream_reshape(src, "ZY", "YZ")
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_nan_inf_through_a_list_selection_uses_written_tolerance():
    src = np.array([np.nan, 1.0, np.inf], dtype=np.float64).reshape(3, 1)
    got = mojo_aicsimageio.reshape_data(src, "ZY", "ZY", Z=[2, 0, 2])
    assert got.shape == (3, 1)
    np.testing.assert_allclose(got, src[[2, 0, 2]], rtol=1e-6, atol=1e-6, equal_nan=True)
    if ref.REFERENCE is not None:
        expected = ref.upstream_reshape(src, "ZY", "ZY", Z=[2, 0, 2])
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_empty_and_one_element_inputs_match_upstream_with_written_tolerance():
    for data, given, ret, kw in [
        (np.zeros((0, 3, 4), dtype=np.float32), "ZYX", "YXZ", {}),
        (np.zeros((0, 0, 0), dtype=np.float32), "ZYX", "YXZ", {}),
        (np.array([7.5], dtype=np.float32).reshape(1, 1), "ZY", "YZ", {}),
        (np.array([7.5], dtype=np.float32), "Z", "Z", {}),
    ]:
        got = mojo_aicsimageio.reshape_data(data, given, ret, **kw)
        if ref.REFERENCE is not None:
            expected = ref.upstream_reshape(data, given, ret, **kw)
            assert got.shape == expected.shape
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
        fallback = _numpy_reshape(data, given, ret, **kw)
        assert got.shape == fallback.shape
        np.testing.assert_allclose(got, fallback, rtol=1e-6, atol=1e-6)


def test_transpose_empty_and_one_element_match_upstream():
    for data, given, ret in [
        (np.zeros((0, 3, 4), dtype=np.uint16), "ZYX", "YXZ"),
        (np.array([42], dtype=np.uint16).reshape(1, 1), "ZY", "YZ"),
        (np.arange(5, dtype=np.uint16), "Z", "Z"),
    ]:
        got = mojo_aicsimageio.transpose_to_dims(data, given, ret)
        expected = expected_transpose(data, given, ret)
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


def test_mismatched_shapes_raise_the_same_class_as_upstream():
    # Upstream: aicsimageio/transforms.py reshape_data checks
    # `check_selection_max > data.shape[dim_index]` then `data[tuple(dim_specs)]`
    # (NumPy `IndexError: tuple index out of range`).
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    for given, ret in [("ZYX", "ZYX"), ("ZYX", "YX")]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, given, ret)
            with pytest.raises(Exception) as ours:
                mojo_aicsimageio.reshape_data(data, given, ret)
            assert type(ours.value).__name__ == type(upstream.value).__name__
        with pytest.raises(IndexError):
            mojo_aicsimageio.reshape_data(data, given, ret)


def test_given_dims_longer_than_rank_matches_upstream_class():
    # Same upstream NumPy getitem that the test above cites.
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    if ref.REFERENCE is not None:
        with pytest.raises(Exception) as upstream:
            ref.upstream_reshape(data, "ZYXW", "YX")
        assert type(upstream.value).__name__ == "IndexError"
    with pytest.raises(IndexError):
        mojo_aicsimageio.reshape_data(data, "ZYXW", "YX")


def test_duplicate_given_dims_match_upstream_conflicting_error():
    # Upstream: aicsimageio/transforms.py reshape_data ends with
    # `return transpose_to_dims(data, given_dims=new_dims, return_dims=return_dims)`
    # (line ~272), and transpose_to_dims (lines ~66-72) raises
    # ConflictingArgumentsError when Counter(given_dims) != Counter(return_dims)
    # or max count > 1. The port must raise the same class here, not read
    # out of bounds.
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    for given, ret in [("ZZ", "ZZ"), ("ZZ", "Z"), ("ZZ", "")]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, given, ret)
            assert type(upstream.value).__name__ in (
                "ConflictingArgumentsError",
                "ValueError",
            )
            with pytest.raises(Exception) as ours:
                mojo_aicsimageio.reshape_data(data, given, ret)
            assert type(ours.value).__name__ == type(upstream.value).__name__
        else:
            with pytest.raises((ConflictingArgumentsError, ValueError)):
                mojo_aicsimageio.reshape_data(data, given, ret)


def test_duplicate_return_dims_match_upstream_conflicting_error():
    # Same upstream reshape_data -> transpose_to_dims check as above:
    # return_dims with a repeated new dimension makes
    # Counter(new_dims) != Counter(return_dims).
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    for given, ret in [("ZY", "TT"), ("ZY", "ZZY")]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, given, ret)
            assert type(upstream.value).__name__ == "ConflictingArgumentsError"
        with pytest.raises(ConflictingArgumentsError):
            mojo_aicsimageio.reshape_data(data, given, ret)


def test_zero_d_result_matches_upstream_value_error():
    # Upstream: transpose_to_dims computes max(Counter(given_dims).values());
    # with new_dims == "" that is `ValueError: max() iterable argument is
    # empty` (transforms.py lines ~66-72 via the reshape_data -> transpose
    # tail). Dropping every axis to 0-d, and 0-d to 0-d, hit that path.
    for data, given, ret, kw in [
        (np.arange(5, dtype=np.uint16), "Z", "", {"Z": 2}),
        (np.arange(5, dtype=np.uint16), "Z", "", {}),
        (np.array(7, dtype=np.uint16), "", "", {}),
    ]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, given, ret, **kw)
            assert type(upstream.value).__name__ == "ValueError"
        with pytest.raises(ValueError, match="max\\(\\) iterable"):
            mojo_aicsimageio.reshape_data(data, given, ret, **kw)


def test_zero_d_new_dim_matches_upstream():
    # 0-d to a new dimension: upstream inserts the dimension and transposes
    # (transforms.py lines ~265-274). The port must return the same value.
    data = np.array(7, dtype=np.uint16)
    got = mojo_aicsimageio.reshape_data(data, "", "T")
    expected = _numpy_reshape(data, "", "T")
    assert got.shape == expected.shape == (1,)
    np.testing.assert_array_equal(got, expected)
    if ref.REFERENCE is not None:
        np.testing.assert_array_equal(got, ref.upstream_reshape(data, "", "T"))


def test_float_string_and_none_kwargs_match_upstream_error_classes():
    # Upstream: a retained-axis spec that is neither int, list/tuple/range
    # nor slice leaves `check_selection_max` unbound and dies with
    # UnboundLocalError (transforms.py lines ~206-224, the `isinstance`
    # ladder). A dropped-axis float reaches `data[tuple(dim_specs)]` and
    # dies there with IndexError.
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    for kw in [{"Z": 1.5}, {"Z": "bad"}, {"Z": None}, {"Z": np.int64(1)}]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, "ZY", "YZ", **kw)
            with pytest.raises(Exception) as ours:
                mojo_aicsimageio.reshape_data(data, "ZY", "YZ", **kw)
            assert type(ours.value).__name__ == type(upstream.value).__name__
        else:
            with pytest.raises(Exception):
                mojo_aicsimageio.reshape_data(data, "ZY", "YZ", **kw)
    for kw in [{"Z": 1.5}, {"Z": 1.0}]:
        if ref.REFERENCE is not None:
            with pytest.raises(Exception) as upstream:
                ref.upstream_reshape(data, "ZY", "Y", **kw)
            assert type(upstream.value).__name__ == "IndexError"
        with pytest.raises(IndexError):
            mojo_aicsimageio.reshape_data(data, "ZY", "Y", **kw)


def test_dropped_axis_string_kwarg_matches_upstream_type_error():
    # Upstream: `check_selection_max = dim_spec` (a str) then
    # `if check_selection_max > data.shape[...]` raises
    # `TypeError: '>' not supported between instances of 'str' and 'int'`
    # (transforms.py lines ~241, ~251).
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    if ref.REFERENCE is not None:
        with pytest.raises(Exception) as upstream:
            ref.upstream_reshape(data, "ZY", "Y", Z="x")
        assert type(upstream.value).__name__ == "TypeError"
    with pytest.raises(TypeError):
        mojo_aicsimageio.reshape_data(data, "ZY", "Y", Z="x")


def test_reduce_to_slice_degenerate_inputs_match_upstream():
    # Upstream: aicsimageio/transforms.py reduce_to_slice lines ~19-32.
    # Empty list dies on steps[0] with IndexError; NaN/inf and repeated
    # steps are handed back untouched.
    if ref.REFERENCE is not None:
        with pytest.raises(Exception) as upstream:
            ref.upstream_reduce_to_slice([])
        assert type(upstream.value).__name__ == "IndexError"
    with pytest.raises(IndexError):
        mojo_aicsimageio.reduce_to_slice([])
    for values in [[2, 2, 2], [5, 3, 1], [float("nan"), 1], [float("inf"), 1]]:
        got = mojo_aicsimageio.reduce_to_slice(list(values))
        if ref.REFERENCE is not None:
            assert repr(got) == repr(ref.upstream_reduce_to_slice(list(values)))
        assert repr(got) == repr(values)
