# mojo-aicsimageio

A Mojo port of the compute surface of
[`aicsimageio`](https://github.com/AllenInstitute/aicsimageio) 4.14.0.

**Short version: `aicsimageio` has no numeric core.** It is a microscopy image
IO library — readers for Bio-Formats, CZI, LIF, ND2, DV, OME-TIFF and
OME-Zarr, writers for OME-TIFF and OME-Zarr, and OME metadata parsing. There is
not a single arithmetic loop over pixel values anywhere in its 12,000 lines
(`aics_image.py`, `transforms.py`, `readers/*`, `writers/*`, `metadata/*`). I
grepped for exactly that before writing anything; what I found were
`for i in range(len(...))` loops over scene and channel *names*, and
`data[i].size * data[i].itemsize` in a TIFF size estimator.

The one place the package moves pixels at array scale is
`aicsimageio/transforms.py`: the dimension engine that reshapes a reader's
stack into whatever dimension order a user asked for. That is what this port
compiles into a single Mojo gather.

**And it is slower than NumPy. The benchmark below reports 0/7 cases at or
above NumPy, geometric mean 0.37x.** That is a real loss and it is discussed in
[Benchmarks](#benchmarks), not buried.

---

## What is ported

| Upstream surface | Status | Notes |
| --- | --- | --- |
| `transforms.reshape_data` | **ported** | Full argument handling transcribed; pixel movement is one Mojo call |
| `transforms.transpose_to_dims` | **ported** | Dimension counter/duplicate checks transcribed; permutation is one Mojo call |
| `transforms.reduce_to_slice` | **ported** | Pure Python in upstream too, so no kernel; behaviour is exact |
| `aicsimageio.exceptions` (4 types) | **ported** | Local definitions, so the package imports without upstream |
| `AICSImage`, all `readers/*`, all `writers/*`, `metadata/*`, `formats.py`, `dimensions.py`, `types.py`, `utils/*` | **not ported** | IO, metadata parsing, Bio-Formats JNI, dask/xarray plumbing. No numeric core, and the numeric core that exists in this package is `transforms`. |

### The kernel

`src/kernels.mojo` exports one symbol, `aics_gather`, which copies `n_out`
whole array elements from `src` to `dst` under a precomputed plan. The plan
describes a gather: every output axis reads a chosen set of indices from a
source axis, and an axis may read *several* source axes at once when NumPy
collapses two or more list selections into its advanced-index block.

Two loops implement it:

* **Run path.** When the fastest-varying output axis walks consecutive source
  elements, the whole transform is a sequence of contiguous runs. The kernel
  copies each run sequentially and advances the odometer once per run. This
  path is specialised per element width (1, 2, 4, 8 bytes) because a width
  test inside the inner loop is loop-invariant but still keeps the body
  scalar, and a nested byte loop in the fallback arm stops the vectoriser
  outright.
* **Gather path.** Otherwise an incremental multi-dimensional odometer drives
  an index-driven copy, O(1) amortised per element rather than one integer
  division per axis.

Elements are copied whole and uninterpreted, so the result is **bit-exact for
every dtype** — the tests use `assert_array_equal`, not `assert_allclose`,
and that is the correct contract here because there is no arithmetic to be
within a tolerance of.

## What is left to the real package

Everything else. Import the real `aicsimageio` alongside this one; they do not
collide (`mojo_aicsimageio` never imports `aicsimageio`).

## Two deliberate differences from upstream

1. **Return value.** Upstream returns a *view* when every selection is a basic
   slice; `ndarray.transpose` is a view. This port always returns a fresh
   C-contiguous array, because the Mojo kernel writes into a caller-owned
   destination. This is the main reason for the benchmark loss: for a dimension
   reorder that leaves the data contiguous, upstream copies nothing.
2. **Array types.** Upstream accepts dask and xarray arrays. This port accepts
   `numpy.ndarray` only and raises `TypeError` naming the type it got.

## Behaviours reproduced verbatim, including upstream's bugs

Parity here is byte-for-byte, error-for-error, so a few upstream rough edges are
preserved on purpose and are covered by tests:

* `reshape_data(data, "ZYX", "YX", Z=slice(None, None, 2))` raises
  `TypeError: bad operand type for abs(): 'NoneType'`. Upstream computes
  `max([abs(dim_spec.stop), abs(dim_spec.start)])` without guarding `None`.
  A bounded slice such as `slice(0, -1, 5)` — the form the upstream docstring
  uses — works fine.
* With two or more list selections, upstream's own `new_dims` bookkeeping
  cannot describe the collapsed rank, so `ndarray.transpose` raises
  `ValueError: axes don't match array`. The port reaches the same `ValueError`.
  The kernel's multi-axis *group* support is therefore only reachable through
  the kernel API, and is tested there directly.
* A `given_dims` string shorter than the array rank survives the selection and
  dies in the transpose with `ValueError`, because NumPy applies an index
  tuple to the leading axes only. The port reproduces that rather than
  validating up front.

## Install and use

The package is pure Python over a compiled `.so`; no build step at import time.

```bash
cd /nvme0n1-disk/code/mojo-aicsimageio
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh                       # -> dist/libmojo-aicsimageio.so
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
PYTHONPATH=python python bench/bench.py
```

```python
import numpy as np, mojo_aicsimageio

stack = np.arange(3 * 4 * 8 * 8, dtype=np.uint16).reshape(3, 4, 8, 8)
plane = mojo_aicsimageio.reshape_data(stack, "ZYXC", "YXC", Z=2)
plane.shape            # (8, 8, 4)
```

`mojo_aicsimageio.kernels()` lists the exported kernel symbols.

## Tests

```
119 passed
```

Two files:

* `tests/test_kernel.py` — the compiled gather directly. Every element width
  including the generic 3/6/16-byte path; SIMD-straddling lengths
  (n = 1003, 1005, 1007, 1009, 1011, 1, 2, 7) so a dropped tail cannot hide;
  reversed and non-monotonic index tables; `src_base`; a zero-extent axis; and
  **a buffer mapped by the OS above 4 GiB**, because a ctypes `argtypes` list
  one entry short is zero-filled, truncates a 64-bit address, and passes every
  small-array test before segfaulting.
* `tests/test_transforms.py` — parity against the real
  `aicsimageio.transforms`. 27 transform cases and 10 error cases, each
  compared for result *and* for exception type.

`aicsimageio` is not in the shared test venv, so `tests/reference.py` resolves
it in this order: already installed, then an extraction cached in
`tests/.refcache/`, then a fresh wheel download from PyPI. The extracted
copy's top-level `__init__.py` is emptied, because `aicsimageio.transforms`
needs only `dask`, `numpy`, `xarray` and the package's own `types`,
`exceptions`, `dimensions` and `image_container` modules — not the long tail of
optional readers. On failure of all three routes the suite falls back to an
independent NumPy reference in `_numpy_reshape`, which is **exercised on every
run regardless**, so a green suite never depends on the download succeeding.
`test_reference_kind_is_reported` prints which reference was live; in the
recorded run it was `wheel`, i.e. real upstream.

**Mutation testing.** The suite was checked against 19 injected bugs — an
odometer that never decrements, an off-by-one at the axis wrap, a dropped
per-axis offset carry, a group table offset by one, a lost source stride, a
reversed transpose, a new axis appended instead of inserted, an off-by-one in
the conflict check, `reduce_to_slice` dropping its step, a `src_base`
double-count, trailing axes collapsed to length 1, a run loop advancing the
destination by one element, and more. All produced named failures. Three
surviving mutants were examined and are not defects: two are algebraically
equivalent to the original (`run_base` expressed without its cancelling terms,
and a `src_base` form), and one only changes which loop is selected, not the
result. The path they left untested — a `given_dims` shorter than the array
rank — now has its own test.

## Benchmarks

`bench/bench.py` gates on correctness first: every case is compared to NumPy
with `array_equal` before it is timed, and a case that fails is reported and
not benchmarked. Best of 15.

The NumPy baseline is the fastest reasonable way to materialise the same
transform: `ndarray.__getitem__` for the selections (which handles strided
slices and list indices natively), `ndarray.transpose` for the reorder, then
materialise in C order. Forcing that last copy is already *unfavourable to
NumPy*, because upstream can return a view and copy nothing. Measured against
upstream's actual behaviour the gap is wider still.

| case | Mojo | NumPy | speedup |
| --- | ---: | ---: | ---: |
| `CZYX -> BSTCZYX` reorder, uint16 2x3x512x512 | 1.23 ms | 0.25 ms | **0.21x** |
| `CZYX -> ZYXC` reorder, uint8 2x3x1024x1024 | 46.49 ms | 36.01 ms | **0.77x** |
| 5D `SCTYZ -> SZCTY`, uint16 64x2x3x128x128 | 42.39 ms | 12.06 ms | **0.28x** |
| strided `Z` selection, uint16 64x3x256x256 | 6.94 ms | 4.45 ms | **0.64x** |
| list `Z` selection, uint16 64x3x256x256 | 1.55 ms | 0.66 ms | **0.43x** |
| drop a `Z` plane and reorder, uint16 64x3x256x256 | 0.35 ms | 0.05 ms | **0.14x** |
| reverse `C` order, float32 2x3x512x512 | 12.60 ms | 6.44 ms | **0.51x** |
| **geometric mean** | | | **0.37x** |

**0 of 7 cases reach NumPy.** This port is a loss and there is no way to read
that table otherwise.

Why:

* This is a permutation of a contiguous buffer. NumPy's `transpose` and
  `__getitem__` are hand-tuned, cache-blocked C. A gather expressed as "read
  index table, copy element" is latency-bound, not bandwidth-bound, and no
  amount of tidying the inner loop changes that.
* The two obvious wins were taken and are visible in the numbers: the
  contiguous-run path lifted the `CZYX -> BSTCZYX` case from 12.79 ms to
  1.23 ms, and the per-width specialisation was needed because the width test
  itself was blocking vectorisation. Neither closes the gap, because NumPy is
  not doing the naive thing either.
* The remaining gap would need cache blocking for the transposing cases (the
  classic shared-`t`-buffer transpose) and explicit SIMD loads in the run
  copy. Neither was attempted; both are real work and the payoff on a workload
  NumPy already handles at memory speed is questionable.

The honest summary: this package's one numeric operation is data movement that
NumPy already does better than a straightforward compiled kernel. The port is
worth having for the exact-match semantics, the single-pass fusion of
selection + padding + reordering into one call, and a single documented entry
point; it is not worth having for speed.

## Layout

```
README.md
build/build.sh              src/kernels.mojo -> dist/libmojo-aicsimageio.so
src/kernels.mojo            one compilation unit, one exported symbol
python/mojo_aicsimageio/
  __init__.py               public API
  transforms.py             reshape_data / transpose_to_dims / reduce_to_slice
  _lib.py                   ctypes bridge, owns every array
  exceptions.py             mirrored from aicsimageio.exceptions
tests/conftest.py
tests/reference.py          resolves the real aicsimageio, or reports that it could not
tests/test_kernel.py        direct kernel tests
tests/test_transforms.py    parity against the real package
bench/bench.py              correctness-gated timing
pixi.toml                   mojo ==1.2.0.dev2026092605
```

Toolchain: `mojo 1.2.0.dev2026092605`, built and run on the shared environment.
No `.pixi` is created and `pixi install` is never run.
