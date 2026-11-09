# VERIFY — independent reproduction of the README claims (second pass)

Independent verifier, no code written by previous phases. Commands run in
order, as specified: `pixi run build && pixi run test`, then `pixi run bench`
(flock-wrapped, best of 15, correctness-gated by `array_equal`).

This is the second VERIFY pass. The first pass (commit `220d2e0`) corrected
the README table to its measured run; this pass re-ran everything from
scratch and treats the first pass's numbers as just another claim.

## 1. Build and tests

`pixi run build && pixi run test` — green. **168 passed**, parity reference
`wheel` (real upstream aicsimageio 4.14.0 `transforms.py`, cached in
`tests/.refcache/`, dask/xarray stubbed). Mojo 1.2.0.dev2026092905 (the
`pixi.toml` pin; README says `mojo ==1.2.0.dev2026092605` in the Layout
section — stale toolchain string, see §4). Mojo deprecation warnings only,
no errors.

### Test parity audit (`git log -p -- tests/`)

Only two commits touch `tests/`: `89838f2` (initial) and `172206d` (phase
test). `git diff 89838f2..HEAD -- tests/` shows **zero removed non-header
lines** — the phase-test commit only *added* cases (CASES 27 → 42, of which
the first 27 are the original matrix; ERROR_CASES 10 → 16; plus degenerate-
input tests and the `reference.py` dask/xarray stub) and never removed an
assertion or loosened a tolerance in an existing test. Nothing to restore.

One related README inaccuracy, not a test change: the README says "40
transform cases and 16 error cases". The error count is right; the transform
count is not — `CASES` now has **42** entries (27 original + 15 degenerate).
The 42 were recounted this session from `tests/test_transforms.py`.

The one tolerance in the suite is `rtol=1e-6, atol=1e-6` with
`assert_allclose` in the three NaN/inf/empty tests (documented in the test
as pinning the exact-copy contract — the kernel has no arithmetic — while
the main matrix uses `assert_array_equal` and exact exception-class equality
against upstream). I checked those three tests against the real upstream
reference directly: they all match upstream exactly, so the tolerance is
not hiding a difference. Independently, this session verified the NumPy
fallback reference (`_numpy_reshape`) agrees with real upstream on all 42
CASES (0 mismatches) and the Mojo port matches upstream on all 7 bench
cases below.

## 2. Benchmark reproduction

`pixi run bench` was run 10 times this session. All 7 cases pass the
correctness gate every run; **0/7 at or above NumPy every run** — the
README's headline ("slower than NumPy, 0/7") reproduces unconditionally.
Geometric mean ranged 0.30x–0.40x across the 10 runs.

Canonical measured run used for the README table (single `pixi run bench`
invocation, best-of-15):

```
CZYX -> BSTCZYX reorder, uint16 3x512x512          0.95ms    0.22ms    0.24x
CZYX -> ZYXC reorder, uint8 3x1024x1024           31.91ms   20.58ms    0.65x
5D SCTYZ -> ZCTY S, uint16 64x3x128x128x3         27.37ms    7.21ms    0.26x
strided Z selection, uint16 64x3x256x256           3.46ms    1.14ms    0.33x
list Z selection, uint16 64x3x256x256              0.92ms    0.51ms    0.55x
drop a Z plane and reorder, uint16 64x3x256x256     0.22ms    0.03ms    0.16x
reverse C order, float32 3x512x512                 7.43ms    3.44ms    0.46x
0/7 cases at or above NumPy; geometric mean speedup 0.34x
```

### Row-by-row vs the current (first-pass) README (20% rule)

Sweeps like rows 2 and 4 below move a lot run-to-run (see the spread note),
so a single canonical run is quoted; the verdicts below are against that
run, and every row was replaced wholesale with it so the table is one
measured run, not a mix.

| case | README (pass 1) | measured | diff | verdict |
|---|---|---|---|---|
| CZYX->BSTCZYX Mojo | 0.95 ms | 0.95 ms | 0% | reproduced |
| CZYX->BSTCZYX NumPy | 0.23 ms | 0.22 ms | −4% | reproduced |
| CZYX->BSTCZYX speedup | 0.24x | 0.24x | 0% | reproduced |
| CZYX->ZYXC Mojo | 32.89 ms | 31.91 ms | −3% | reproduced |
| CZYX->ZYXC NumPy | 19.96 ms | 20.58 ms | +3% | reproduced |
| CZYX->ZYXC speedup | 0.61x | 0.65x | +7% | reproduced |
| 5D SCTYZ Mojo | 28.33 ms | 27.37 ms | −3% | reproduced |
| 5D SCTYZ NumPy | 7.63 ms | 7.21 ms | −6% | reproduced |
| 5D SCTYZ speedup | 0.27x | 0.26x | −4% | reproduced |
| strided Z Mojo | 3.73 ms | 3.46 ms | −7% | reproduced |
| strided Z NumPy | 1.31 ms | 1.14 ms | −13% | reproduced |
| strided Z speedup | 0.35x | 0.33x | −6% | reproduced |
| list Z Mojo | 0.89 ms | 0.92 ms | +3% | reproduced |
| list Z NumPy | 0.56 ms | 0.51 ms | −9% | reproduced |
| list Z speedup | 0.63x | 0.55x | −13% | reproduced |
| drop Z Mojo | 0.21 ms | 0.22 ms | +5% | reproduced |
| drop Z NumPy | 0.03 ms | 0.03 ms | 0% | reproduced |
| drop Z speedup | 0.16x | 0.16x | 0% | reproduced |
| reverse C Mojo | 7.50 ms | 7.43 ms | −1% | reproduced |
| reverse C NumPy | 3.40 ms | 3.44 ms | +1% | reproduced |
| reverse C speedup | 0.45x | 0.46x | +2% | reproduced |
| geometric mean | 0.35x | 0.34x | −3% | reproduced |

No figure moved more than 20%, so no row *required* correction; the table
was still replaced with this session's measured run (the point of the
exercise is that every number in it was measured in *this* session), which
changed 13 of the 21 figures by ≤13% and the geometric mean 0.35x → 0.34x.
No row was removed or reworded: all 7 stay slower than NumPy, the 0/7 and
geometric-mean headline lines are kept. (Against the pre-pass-1 numbers,
46.49 ms etc., the first pass's corrections stand — those were never
reproducible and this pass's runs confirm they are not the numbers this
host produces.)

Note on run-to-run spread: `strided Z` NumPy time ranged 1.14–3.02 ms
across my 10 runs (its `__getitem__` path is noisy on this host), so its
speedup ranged 0.32x–0.72x — always <1x, but the single-run 0.33x should
not be quoted to two significant figures. `CZYX -> BSTCZYX` Mojo ranged
0.90–1.56 ms (speedup 0.14x–0.33x). All other cases are stable within
~10%. The honest claim is "0/7, geometric mean roughly a third of NumPy".

### Same-input / no-skipped-work check

`bench/bench.py` CASES: one `data` array per case (`default_rng(1234)` per
case), timed through `reshape_data(data, ...)` and `_numpy_reshape(data, ...)`
with identical `given`/`ret`/`kwargs`. Correctness gate compares both to the
same expected values with `array_equal` before timing; all 7 match, and this
session independently re-checked the 7 bench cases against the real upstream
`transforms.reshape_data` — all equal, all C-contiguous on the Mojo side.

- **Mojo side** allocates `dst = np.empty(n_out)` and the kernel writes all
  `n_out` elements — no skipping, no view. Timing includes full Python
  argument handling, plan building and the ctypes call. Two caches exist
  (`_plan` scratch, fully rewritten per call; `_ones_cache` for the run
  detector, keyed by size). The gate call warms both, so the timed calls run
  with warm caches — slightly *pro*-Mojo, and it still loses 0/7.
- **NumPy side** is `data[tuple(key)]` → `transpose` → `np.array(..., order="C")`,
  i.e. a forced materialising copy. For the list-selection case the
  `__getitem__` already copies (advanced indexing) and then `np.array`
  copies again — two copies to Mojo's one — so the baseline is, if anything,
  handicapped there and still wins. No caching or memoisation on either side.
  Same input, same output contract, same output size. The speedup definition
  (`t_np / t_mojo`) is consistent.
- One labelling flaw in the harness, cosmetic only: row 1 is labelled
  `CZYX -> BSTCZYX` but the case as coded is the identity `CZYX -> CZYX`
  ("BSTCZYX" appears only in the label string — `given="CZYX", ret="CZYX"`).
  This session timed the *literal* `CZYX -> BSTCZYX` transform separately:
  Mojo 1.10 ms vs NumPy 0.25 ms, speedup 0.23x — indistinguishable from the
  coded identity case (both are a contiguous copy of the same buffer), so
  the mislabel does not change the conclusion, but the README row 1 should
  be read as "the identity/contiguous-copy case". Likewise bench labels the
  5D case "uint16 64x3x128x128x3" while its shape tuple is
  `(64, 2, 3, 128, 128)`; the README's `64x2x3x128x128` is correct.

## 3. What I changed

- `README.md` benchmark table: replaced with the canonical measured run
  above (no figure moved >20%; 13 figures and the geometric mean 0.35x →
  0.34x changed within tolerance). No rows removed, no rewording, still 0/7.
  Note in passing: the "lifted from 12.79 ms to 0.95 ms" line was *not*
  changed this pass — the 12.79 ms starting point is another phase's number
  that this session did not reproduce (see §4).
- `bench/verify.md` (this file, replacing the first pass's version).
- No code changes: no failing test was found, so per the rules nothing in
  `src/` or `python/` was touched.

## 4. What I could not reproduce

- The pre-pass-1 absolute numbers (46.49 / 36.01 ms etc.) — not reproducible
  on this host; pass 1's corrections stand.
- The "12.79 ms" pre-optimisation figure for `CZYX -> BSTCZYX` in the
  README's Why section — no artefact in this repo reproduces it; it is an
  earlier phase's number and I did not measure it. Left in place (it is
  historical context, not a benchmark-table cell), but it is **not** a
  number measured in either VERIFY session.
- `bench/analysis.md`'s numbers (e.g. 34.73 ms for CZYX->ZYXC) — measured
  under a different setup; close to but not identical to this session's runs.
- The README's "40 transform cases" (now 42) and `mojo ==1.2.0.dev2026092605`
  in the Layout section (pixi.toml pins `1.2.0.dev2026092905`, and
  `pixi run mojo --version` reports `Mojo 1.2.0.dev2026092905`). Both are
  stale strings; the toolchain one was outside the benchmark table so I did
  not edit it — flagging here instead.
- The "168 passed" test count reproduces exactly.
