# VERIFY — independent reproduction of the README claims

Independent verifier, no code written by previous phases. Commands run in order,
as specified: `pixi run build && pixi run test`, then `pixi run bench` (flock-wrapped,
best of 15, correctness-gated by `array_equal`).

## 1. Build and tests

`pixi run build && pixi run test` — green. 168 passed, parity reference
`wheel` (real upstream aicsimageio 4.14.0 `transforms.py`, cached in
`tests/.refcache/`, dask/xarray stubbed). Mojo deprecation warnings only,
no errors.

Test parity audit (`git log -p -- tests/`): only two commits touch `tests/`
(`89838f2` initial, `172206d` phase test). `git diff 89838f2..HEAD -- tests/`
shows **zero removed lines** — the phase-test commit only *added* cases
(40 transform + 16 error cases, degenerate inputs, `reference.py` dask/xarray
stub) and never removed an assertion or loosened a tolerance in an existing
test. New tests use `rtol=1e-6, atol=1e-6` with `assert_allclose` for the
NaN/inf/empty cases (documented in the test as pinning the exact-copy
contract — the kernel has no arithmetic — while the main matrix still uses
`assert_array_equal` and exact exception-class equality against upstream).
Nothing to restore.

## 2. Benchmark reproduction

Ran `pixi run bench` multiple times. All 7 cases pass the correctness gate
every run; geometric mean 0.33–0.37x across runs, 0/7 at or above NumPy
every run — the README's headline ("slower than NumPy, 0/7") reproduces.

Canonical measured run used for the README table (single `pixi run bench`
invocation, best-of-15):

```
CZYX -> BSTCZYX reorder, uint16 3x512x512          0.95ms     0.23ms    0.24x
CZYX -> ZYXC reorder, uint8 3x1024x1024           32.89ms    19.96ms    0.61x
5D SCTYZ -> ZCTY S, uint16 64x3x128x128x3         28.33ms     7.63ms    0.27x
strided Z selection, uint16 64x3x256x256           3.73ms     1.31ms    0.35x
list Z selection, uint16 64x3x256x256              0.89ms     0.56ms    0.63x
drop a Z plane and reorder, uint16 64x3x256x256     0.21ms     0.03ms    0.16x
reverse C order, float32 3x512x512                 7.50ms     3.40ms    0.45x
0/7 cases at or above NumPy; geometric mean speedup 0.35x
```

### Row-by-row vs the old README (20% rule)

| case | old | measured | diff | verdict |
|---|---|---|---|---|
| CZYX->BSTCZYX Mojo | 1.23 ms | 0.95 ms | −23% | **corrected** |
| CZYX->BSTCZYX NumPy | 0.25 ms | 0.23 ms | −8% | within 20%, corrected anyway with the row |
| CZYX->BSTCZYX speedup | 0.21x | 0.24x | +14% | within 20% |
| CZYX->ZYXC Mojo | 46.49 ms | 32.89 ms | −29% | **corrected** |
| CZYX->ZYXC NumPy | 36.01 ms | 19.96 ms | −45% | **corrected** |
| CZYX->ZYXC speedup | 0.77x | 0.61x | −21% | **corrected** |
| 5D SCTYZ Mojo | 42.39 ms | 28.33 ms | −33% | **corrected** |
| 5D SCTYZ NumPy | 12.06 ms | 7.63 ms | −37% | **corrected** |
| 5D SCTYZ speedup | 0.28x | 0.27x | −4% | within 20% |
| strided Z Mojo | 6.94 ms | 3.73 ms | −46% | **corrected** |
| strided Z NumPy | 4.45 ms | 1.31 ms | −71% | **corrected** |
| strided Z speedup | 0.64x | 0.35x | −45% | **corrected** |
| list Z Mojo | 1.55 ms | 0.89 ms | −43% | **corrected** |
| list Z NumPy | 0.66 ms | 0.56 ms | −15% | within 20% |
| list Z speedup | 0.43x | 0.63x | +47% | **corrected** |
| drop Z Mojo | 0.35 ms | 0.21 ms | −40% | **corrected** |
| drop Z NumPy | 0.05 ms | 0.03 ms | −40% | **corrected** |
| drop Z speedup | 0.14x | 0.16x | +14% | within 20% |
| reverse C Mojo | 12.60 ms | 7.50 ms | −41% | **corrected** |
| reverse C NumPy | 6.44 ms | 3.40 ms | −47% | **corrected** |
| reverse C speedup | 0.51x | 0.45x | −12% | within 20% |
| geometric mean | 0.37x | 0.35x | −5% | within 20%, updated to 0.35x |

Every row whose figure moved >20% was replaced with the measured number; rows
within 20% were replaced too so the table is one measured run, not a mix.
No row was removed or reworded: all 7 stay slower than NumPy, the 0/7 and
geometric-mean lines are kept. The old numbers look like an older/slower
machine or an earlier kernel — the current box is faster on both sides,
and NumPy got faster by more, which is why the CZYX->ZYXC and strided
speedups moved the most.

Note on run-to-run spread: `strided Z` NumPy time varies 1.1–2.8 ms across
runs of this script (its `__getitem__` path is noisy on this host), so its
speedup varies 0.31–0.70x run to run — always <1x, but the single-run
0.35x should not be quoted to two significant figures. All other cases are
stable within ~10%.

## 3. Same-input / no-skipped-work check

`bench/bench.py` CASES: one `data` array per case (`default_rng(1234)` per
case), timed through `reshape_data(data, ...)` and `_numpy_reshape(data, ...)`
with identical `given`/`ret`/`kwargs`. Correctness gate compares both to the
same expected values with `array_equal` before timing; all 7 match, and I
independently re-checked the 7 bench cases against the real upstream
`transforms.reshape_data` — all equal.

- **Mojo side** allocates `dst = np.empty(n_out)` and the kernel writes all
  `n_out` elements — no skipping, no view. Timing includes full Python
  argument handling, plan building and the ctypes call.
- **NumPy side** is `data[tuple(key)]` → `transpose` → `np.array(..., order="C")`,
  i.e. a forced materialising copy. It does not skip work either; if
  anything it is *advantaged* relative to upstream (which for basic-slice
  cases returns a view and copies nothing — the README documents this, and
  the baseline is deliberately stricter than upstream).
- No caching or memoisation on either side; `_plan` is reused as scratch
  but fully rewritten per call. Same input, same output contract, same
  output size. The speedup definition (`t_np / t_mojo`) is consistent.

One cosmetic label bug found, not affecting timing: `bench.py` labels the
5D case "uint16 64x3x128x128x3" while its shape tuple is `(64, 2, 3, 128, 128)`
and the README correctly says `64x2x3x128x128`. Left as-is — labels are not
numbers and the README (the corrected artefact) is right.

## 4. What I changed

- `README.md` benchmark table: replaced with the measured run above
  (6 of 7 Mojo times, 5 of 7 NumPy times, 3 speedups and the geometric mean
  moved >20% or tracked their rows). Headline geometric mean 0.37x → 0.35x,
  and the "lifted from 12.79 ms to 1.23 ms" line → 0.95 ms. No rows removed,
  no rewording, still 0/7.
- `bench/verify.md` (this file).
- No code changes: no failing test was found, so per the rules nothing in
  `src/` or `python/` was touched.

## 5. What I could not reproduce

- The old absolute numbers (46.49 / 36.01 ms etc.) — not reproducible on
  this host today, as above; no way to know if they were a different machine
  or an earlier kernel build. Wrong going forward, hence corrected.
- `bench/analysis.md`'s numbers (e.g. 34.73 ms for CZYX->ZYXC) are from a
  different measurement setup and were *not* re-verified here; they are
  close to but not identical to this run.
- The "168 passed" test count reproduces exactly.
