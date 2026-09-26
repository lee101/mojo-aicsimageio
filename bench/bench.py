"""Correctness-gated benchmark for the Mojo N-D gather.

Every case is checked against NumPy before it is timed; a case that does not
match exactly is reported as FAILED and not benchmarked. The kernel only moves
whole array elements, so the correctness gate is exact equality -- there is no
arithmetic to be within a tolerance of.

The NumPy baseline is the fastest reasonable way to materialise the same
transform without Mojo: apply the per-axis selections with `ndarray.__getitem__`
(which handles strided slices and list indices natively), apply the
dimension-order change with `ndarray.transpose`, and materialise the result in
C order. Forcing that last copy matters: upstream `aicsimageio` returns a
*view* whenever a dimension reorder leaves the data contiguous, so it copies
nothing at all in those cases. This port always returns a fresh array, so the
baseline is held to the same contract here; measured against upstream's
view-returning behaviour the gap is wider still, and the README says so.
"""

import sys
import time

import numpy as np

import mojo_aicsimageio

REPEATS = 15


def _time(fn):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def _numpy_reshape(data, given, ret, **kw):
    key = []
    for dim in given:
        spec = kw.get(dim)
        if dim not in ret:
            key.append(0 if spec is None else spec)
        elif spec is None:
            key.append(slice(None))
        elif isinstance(spec, int):
            key.append(slice(spec, spec + 1))
        else:
            key.append(spec)
    result = data[tuple(key)]
    new_dims = given
    for dim in given:
        if dim not in ret:
            new_dims = new_dims.replace(dim, "")
    for i, dim in enumerate(ret):
        if dim not in given:
            new_dims = f"{new_dims[:i]}{dim}{new_dims[i:]}"
            result = result.reshape(*result.shape[:i], 1, *result.shape[i:])
    return np.array(
        result.transpose([new_dims.find(d) for d in ret]), order="C"
    )


CASES = [
    # (label, shape, dtype, given, ret, kwargs)
    (
        "CZYX -> BSTCZYX reorder, uint16 3x512x512",
        (2, 3, 512, 512),
        np.uint16,
        "CZYX",
        "CZYX",
        {},
    ),
    (
        "CZYX -> ZYXC reorder, uint8 3x1024x1024",
        (2, 3, 1024, 1024),
        np.uint8,
        "CZYX",
        "ZYXC",
        {},
    ),
    (
        "5D SCTYZ -> ZCTY S, uint16 64x3x128x128x3",
        (64, 2, 3, 128, 128),
        np.uint16,
        "SCTYZ",
        "SZCTY",
        {},
    ),
    (
        "strided Z selection, uint16 64x3x256x256",
        (64, 3, 256, 256),
        np.uint16,
        "ZCYX",
        "ZCYX",
        {"Z": slice(0, 64, 2)},
    ),
    (
        "list Z selection, uint16 64x3x256x256",
        (64, 3, 256, 256),
        np.uint16,
        "ZCYX",
        "ZCYX",
        {"Z": [0, 5, 11, 20, 33, 44, 51, 62]},
    ),
    (
        "drop a Z plane and reorder, uint16 64x3x256x256",
        (64, 3, 256, 256),
        np.uint16,
        "ZCYX",
        "CYX",
        {"Z": 7},
    ),
    (
        "reverse C order, float32 3x512x512",
        (2, 3, 512, 512),
        np.float32,
        "CZYX",
        "CYXZ",
        {},
    ),
]


def main():
    print(f"{'case':46s} {'Mojo':>9s} {'NumPy':>9s} {'speedup':>8s}  result")
    print("-" * 92)
    rows = []
    for label, shape, dtype, given, ret, kw in CASES:
        rng = np.random.default_rng(1234)
        data = (rng.random(shape) * 100).astype(dtype)

        expected = _numpy_reshape(data, given, ret, **kw)
        got = mojo_aicsimageio.reshape_data(data, given, ret, **kw)
        if got.shape != expected.shape or not np.array_equal(got, expected):
            print(f"{label:46s} FAILED correctness gate "
                  f"({got.shape} vs {expected.shape})")
            rows.append((label, None, None, "FAILED"))
            continue

        t_mojo, _ = _time(
            lambda: mojo_aicsimageio.reshape_data(data, given, ret, **kw)
        )
        t_np, _ = _time(lambda: _numpy_reshape(data, given, ret, **kw))
        ratio = t_np / t_mojo
        verdict = "mojo" if ratio >= 1.0 else "numpy"
        print(
            f"{label:46s} {t_mojo * 1e3:8.2f}ms {t_np * 1e3:8.2f}ms "
            f"{ratio:7.2f}x  {verdict}"
        )
        rows.append((label, t_mojo, t_np, ratio))

    good = [r for r in rows if r[1] is not None]
    if not good:
        print("\nno case passed its correctness gate")
        return 1
    wins = [r for r in good if r[3] >= 1.0]
    geo = float(
        np.exp(np.mean([np.log(r[3]) for r in good]))
    )
    print("-" * 92)
    print(
        f"{len(wins)}/{len(good)} cases at or above NumPy; "
        f"geometric mean speedup {geo:.2f}x"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
