# ANALYZE — measured map of where the time goes

`mojo-aicsimageio` (Mojo port of the `aicsimageio.transforms` dimension
engine). ANALYZE phase only: no algorithms changed, nothing in `src/` or
`tests/` touched. Every number below was produced by a run in this session
(commands listed under Method); no figures are estimated.

Hardware: Intel Xeon E5-2697 v4, 72 threads (2 sockets) — same host the
`pixi run bench` (flock-wrapped) task ran on. `aicsimageio` upstream was
resolved through `tests/reference.py` as kind `wheel` (4.14.0, cached in
`tests/.refcache/4.14.0/`); it is not importable directly in the pixi env.
Upstream returns a *view* whenever every selection is a basic slice
(`transpose` is a view); this port always returns a fresh C-contiguous
array. Both comparisons are given so the view-vs-copy gap is not mistaken
for kernel speed.

## Ranked hotspot table

Ranked by measured `reshape_data` / `transpose_to_dims` wall time,
largest first. Times are the `pixi run bench` run (best of 15) for the
seven `bench/bench.py` cases — measured again and reproduced below — and
targeted follow-ups where stated. `bytes touched = 2 × n_out × itemsize`
(read `n_out` elements + write `n_out` elements; scatters that read less
than the full source under-count slightly, which is immaterial to ranking).
`flops = 0`: the kernel is pure data movement (bit-exact element copies),
no floating-point arithmetic. `flops/byte = 0` for every row.

| # | function | input (shape, dtype, op) | time | bytes touched | flops | flops/byte | bound | time sits in | vs upstream |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `reshape_data` | `(2,3,1024,1024)` uint8, CZYX→ZYXC, n_out=6,291,456 | 34.73 ms | 12,582,912 | 0 | 0.00 | memory (gather **latency**-bound; 0.36 GB/s) | kernel (glue 2.7%) | SLOWER (upstream view 0.02 ms; upstream+copy 19.24 ms) |
| 2 | `reshape_data` | `(64,2,3,128,128)` uint16, SCTYZ→SZCTY, n_out=6,291,456 | 27.91 ms | 25,165,824 | 0 | 0.00 | memory (gather latency-bound; 0.90 GB/s) | kernel (glue <4%) | SLOWER (upstream view 0.01 ms; upstream+copy 6.39 ms) |
| 3 | `transpose_to_dims` | `(2,3,512,512)` uint16, CZYX→ZYXC, n=1,572,864 | 8.58 ms | 6,291,456 | 0 | 0.00 | memory (gather latency-bound; 0.73 GB/s) | kernel | SLOWER (same reason) |
| 4 | `reshape_data` | `(2,3,512,512)` float32, CZYX→CYXZ, n=1,572,864 | 7.67 ms | 12,582,912 | 0 | 0.00 | memory (gather latency-bound; 1.64 GB/s) | kernel (glue 3.1%) | SLOWER (upstream view 0.01 ms; upstream+copy 3.49 ms) |
| 5 | `reshape_data` | `(64,3,256,256)` uint16, strided Z=0:64:2, n_out=6,291,456 | 3.83 ms | 25,165,824 | 0 | 0.00 | memory (run path; 6.57 GB/s) | kernel (glue 10.3%) | SLOWER (upstream view 0.01 ms) |
| 6 | `transpose_to_dims` | `(2,3,512,512)` uint16, identity CZYX→CZYX, n=1,572,864 | 1.41 ms | 6,291,456 | 0 | 0.00 | memory (run path; 4.45 GB/s) | kernel + glue (~170 µs) | SLOWER (upstream view 0.01 ms; upstream+copy 0.26 ms) |
| 7 | `reshape_data` | `(2,3,512,512)` uint16, identity CZYX→BSTCZYX, n=1,572,864 | 0.91 ms | 6,291,456 | 0 | 0.00 | memory (run path; 6.91 GB/s) | kernel 81% / glue 19% | SLOWER (upstream view 0.01 ms; upstream+copy 0.26 ms) |
| 8 | `reshape_data` | `(64,3,256,256)` uint16, list Z (8 idx), n_out=1,572,864 | 0.80 ms | 6,291,456 | 0 | 0.00 | memory (run path; 7.86 GB/s) | kernel (glue ~kernel/noise) | SLOWER (upstream actually copies here: 0.27 ms) |
| 9 | `reshape_data` | `(64,3,256,256)` uint16, Z=7 → CYX, n_out=196,608 | 0.21 ms | 786,432 | 0 | 0.00 | memory (run path; 3.74 GB/s) | kernel 55% / glue 45% | SLOWER (upstream view 0.01 ms) |
| 10 | `reduce_to_slice` | list, 8 elements | 1.54 µs / call | 0 pixel bytes (index list only) | 7 integer subtractions | n/a | latency (pure-Python) | Python (no kernel, no ctypes) | — |
| 11 | `kernels()` | introspection | 16.3 µs / call | 0 | 0 | n/a | latency (Python) | Python | — |

Slower-than-upstream: **every pixel-moving row is slower**, both against
upstream-as-contracted (view, 0.01–0.02 ms → 22×–2100× slower, because
upstream copies nothing) and against upstream forced to materialise
(`np.array(...,copy=True,order="C")`: 0.26 / 19.24 / 6.39 / 3.49 ms → Mojo
3.55× / 1.63× / 4.01× / 2.12× slower respectively). The view comparison is
the honest upstream semantic; the forced-copy comparison is the honest
compute comparison. See "Upstream vs NumPy" below. The rows are kept, not
dropped. Every row is also slower than the NumPy materialising baseline
(0/7 at or above NumPy, geometric mean 0.38× in the flock-wrapped run;
NumPy effective bandwidth 2.0–20.7 GB/s vs Mojo 0.3–8.4 GB/s on the same
cases).

Roofline: this host streams contiguous copies at 28.99–40.61 GB/s
(`np.copyto`, 3–12 MB). The run path (rows 5–9) reaches ~4–8 GB/s —
within ~5× of streaming, so vectorisation can still matter. The gather
path (rows 1–4) reaches 0.3–1.7 GB/s — one to two orders below streaming —
because each output element pays an index-table lookup and a dependent
load through the odometer; hot-vs-cold cache eviction on row 4 changed
time by only 8.32 → 8.87 ms, confirming this is latency, not bandwidth,
limitation. Transposes also vary hugely with axis order at fixed size
(`CZYX→CXZY` 15.77 ms vs `CZYX→CZYX` 1.02 ms vs `CZYX→ZYXC` 8.12 ms on
`(2,3,512,512)` uint16): gather locality, not size, sets the time. A
dtype sweep at fixed size/order (`uint8` 3.69 ms, `uint16` 3.77 ms,
`uint32`/`float32` 2.20 ms, `float64` 2.31 ms) shows time tracks
`n_out` (element count), not bytes — again latency-bound.

## Where the time sits: glue vs kernel

Decomposition measured by timing `reshape_data`, `_lib._run` (plan fill +
`_contiguous_run` + one ctypes call + kernel), and a raw `lib.aics_gather`
call on a reused plan, plus `cProfile` (small case: `reshape_data` 12 ms
cumulative over 50 calls; large uint8 case: `_run` 0.328 s of 0.332 s) and
a best-of-19 re-run that reproduced the same ratios at different absolute
times (host variance; ratios stable).

* The kernel dominates at image sizes. Glue share is 2.7% (row 1), ~3%
  (row 4), 10–19% on the mid-size run-path rows, and 45% only on the
  smallest row (row 9, 196 k elements). A ctypes call with `n_out=0`
  costs ~13 µs; `_run` on a prebuilt plan matches raw-lib within noise
  (0.793 vs 0.784 ms row 7; 31.719 vs 31.291 ms row 1), i.e. plan fill +
  `_contiguous_run` are <10 µs and not the problem at scale.
* Per-call overhead floor: tiny `(3,8,8)` uint16 takes 126 µs (Mojo) vs
  5.9 µs (NumPy); `(2,3,64,64)` takes 186 µs vs 11.8 µs (21× / 16×).
  On small arrays the port can never win regardless of kernel quality.
* `_contiguous_run` alone is ~11 µs per call (best-of-30, small case).
  It allocates `np.diff(last_table)` — `(last_size−1)×8` bytes (504 B for
  the 64-wide axis, 8 B for the 2-wide axis) — plus `np.array_equal`,
  on every call. `_selected_axes` is 31 µs (small) / 188 µs (large);
  `_slice_indices` ×4 is 5–34 µs; `_c_strides` ~1 µs; `AxisPlan` ×4 ~4 µs.
  None of these move the large-case ranking, but together with the
  `np.empty` and `np.prod` they are the entire small-array tax.

## Flags

**Python loops that re-enter ctypes per element: none.** Verified by
instrumenting `lib.aics_gather`: one call per `reshape_data` even at
6.29 M output elements (a per-element driver would issue ~6.29 M calls).
The only ctypes crossing is `_lib._run` → `aics_gather` (7 `c_int64` args)
per `reshape_data`/`transpose_to_dims` call.

**Python per-element loops that do exist** (over index axes, never over
pixels — axis extents ≤ 1024 in all measured cases, so second-order at
these sizes, first-order only for small arrays):

* `transforms._list_indices`: Python loop at ~161–289 ns/element
  (2.31 µs for 8 indices, best-of-200). For a 64-wide Z axis that is
  ~10–18 µs of pure Python per call.
* `transforms._slice_indices`: `list(range(...))` in Python — 5.34 µs to
  build a 256-int list (512-wide axis, step 2). Scaling in axis extent,
  hidden inside the `_slice_indices` ×4 figure above.
* `transforms.reduce_to_slice`: list comprehension for deltas; 0.26 µs
  (len 1) to 39.7 µs (len 512). Realistic call sites (row 10) cost
  1.54 µs — not a hotspot unless an axis selection exceeds ~1000 entries.
* `_lib._run` plan fill: Python loop over axes (not elements) + a NumPy
  slice-assign per table; measured at ~23 ns/plan-element incl. kernel at
  N=1024, amortised below noise at image sizes. `_contiguous_run`'s
  `np.diff` is the only per-call allocation proportional to an axis extent.

**Per-call allocations / copies:**

* Unavoidable-by-contract: fresh `dst = np.empty(n_out)` = 6.29 MB at the
  largest cases (tracemalloc: 3.1–6.3 MB per call); this is the entire
  point of the port's divergence from upstream's view.
* Plan index tables: `np.int64` arrays, one per axis, totalling 4.3–16.6 KB
  per call (1052–2076 int64s); the shared `_plan` (4096 int64s) is
  reused and only grows to the high-water mark.
* `AxisPlan` re-wraps every table through `np.ascontiguousarray`
  (no-copy for contiguous int64; copies if a handed-in table is
  non-contiguous or wrong dtype).
* `np.ascontiguousarray(data)` in `_run_plan` is a no-copy alias when the
  input is C-contiguous (0.17–0.25 µs measured), but a **full input copy**
  otherwise: on a `[:, ::2, :, :]` view (4.2 MB logical), the copy alone
  took 0.468 ms of a 2.033 ms call — 23%. Callers passing strided/dask
  views pay an undocumented preprocessing tax.
* `_ones_cache` holds one `ones(size−1)` int64 per distinct
  fastest-axis size for the lifetime of the process (≤ `8×1024` B in
  practice); negligible, but unbounded if callers vary axis sizes widely.

## Next phase per hotspot

All rows have flops/byte = 0 (rows 1–9; rows 10–11 are pure Python with
no kernel). A GPU verdict would need to justify itself from that ratio:
0 flops/byte is far below any GPU break-even (the kernel is latency-bound
and below the ~2 flops/byte CPU-only threshold in `MOJO_NOTES.md` §5), so
GPU is rejected for every row below — each name is justified by the
flops/byte figure, and none relies on "the API is missing" (`max.gpu`
exists per the notes).

| hotspot | next phase | why (evidence) |
|---|---|---|
| Row 1 — uint8 `CZYX→ZYXC` gather (34.73 ms, 0.36 GB/s) | **parallel** | Largest absolute time; output is embarrassingly parallel over runs (72 threads idle, kernel is serial). flops/byte = 0 ⇒ GPU rejected; parallel first (latency amortisation via more outstanding loads), explicit SIMD second. |
| Row 2 — 5D SCTYZ gather (27.91 ms, 0.90 GB/s) | **parallel** | Same structure: long rank-5 odometer, 6.29 M independent outputs, serial kernel. `parallelize` from `max.algorithm` (per MOJO_NOTES §4/5b, explicit worker count + size gate). flops/byte = 0 ⇒ not GPU. |
| Row 3 — `transpose_to_dims` ZYXC gather (8.58 ms, 0.73 GB/s) | **parallel** | Shares row-1 kernel; one fix covers both entry points. Cache-blocked tiling alone would not fix latency at 0.73 GB/s without more outstanding loads. flops/byte = 0 ⇒ not GPU. |
| Row 4 — float32 reverse-C gather (7.67 ms, 1.64 GB/s) | **parallel** | Last axis extent 3 ⇒ tiny runs, worst gather locality per element; only the gather path. Parallel over output blocks; unroll the width-specialised loop second. flops/byte = 0 ⇒ not GPU. |
| Row 5 — strided-Z run path (3.83 ms, 6.57 GB/s) | **simd** | Already on the run path; gap is scalar `for j` copy vs 29–40 GB/s streaming. Explicit `load[width=W]/store` (MOJO_NOTES §3) in `run_copy_*` is the cheapest next step. flops/byte = 0 ⇒ not GPU. |
| Row 6 — `transpose_to_dims` identity run (1.41 ms, 4.45 GB/s) | **simd** | Same run-copy loop as row 7; eliminating the Python-side `_contiguous_run` diff alloc afterward is a glue trim (time <10 µs), not a kernel phase. flops/byte = 0 ⇒ not GPU. |
| Row 7 — identity run path (0.91 ms, 6.91 GB/s) | **simd** | Single contiguous 512-wide run per (C,Z,Y); SIMD + unroll on `run_copy_u16` directly lifts the fastest run row. flops/byte = 0 ⇒ not GPU. |
| Row 8 — list-Z run path (0.80 ms) | **simd** (kernel); **none** for the list handling | Kernel is the same run loop as rows 5–7. `_list_indices` Python loop (~2 µs) is not a kernel-phase hotspot; leave to Python trim. flops/byte = 0 ⇒ not GPU. |
| Row 9 — dropped-Z-plane (0.21 ms, glue 45%) | **none** | No kernel phase repays here: 196 k outputs, 171 µs of glue (axis/table/plan/ctypes) dwarfs any vectorisation. Do not optimise the kernel for this row. flops/byte = 0 ⇒ not GPU. |
| Row 10 — `reduce_to_slice` | **none** | Pure-Python upstream-faithful helper; 1.54 µs realistic. No Mojo kernel exists to accelerate; not on any image-scale path. |
| Row 11 — `kernels()` | **none** | Introspection only; 16 µs. |

Unroll alone is not recommended as a first step for the gather rows
(1–4): the odometer dependency already defeats ILP, so unrolling without
more outstanding loads changes nothing measurable. SIMD is recommended
before unroll on the run rows (5–8) because the run body is proven
sequential (rows 5–8 sit on the run path at 4–8 GB/s and the source has
zero `load[width=]` sites — measured by grep: 0 occurrences).

## Upstream vs NumPy (measured this session)

`pixi run bench` (best of 15; the exact `bench/bench.py` baseline, which
forces a C-order materialise so NumPy copies like the port):

```
case                                                Mojo     NumPy  speedup
CZYX -> BSTCZYX reorder, uint16 3x512x512          0.91ms     0.21ms    0.23x
CZYX -> ZYXC reorder, uint8 3x1024x1024           34.73ms    20.67ms    0.60x
5D SCTYZ -> ZCTY S, uint16 64x3x128x128x3         27.91ms     8.01ms    0.29x
strided Z selection, uint16 64x3x256x256           3.83ms     2.53ms    0.66x
list Z selection, uint16 64x3x256x256              0.80ms     0.53ms    0.65x
drop a Z plane and reorder, uint16 64x3x256x256    0.21ms     0.03ms    0.16x
reverse C order, float32 3x512x512                 7.67ms     3.38ms    0.44x
0/7 at or above NumPy; geometric mean 0.38x
```

Upstream `aicsimageio.transforms.reshape_data` (4.14.0 wheel, best of 15;
returns views for 6 of 7 cases) vs Mojo vs upstream-forced-copy:

```
case                 Mojo     NumPy      upstream       upstream+copy
identity uint16      0.92ms   0.23ms     0.01ms view   0.26ms
ZYXC uint8           31.39ms  18.82ms    0.02ms view   19.24ms
5D SCTYZ             24.40ms  6.82ms     0.01ms view   6.39ms
strided Z            3.71ms   1.86ms     0.01ms view   —
list Z copy          0.91ms   0.52ms     0.27ms copy   —
drop Z               0.23ms   0.04ms     0.01ms view   —
reverse C float32    7.44ms   3.32ms     0.01ms view   3.49ms
```

Read: upstream-as-contracted is 22×–2100× faster because it moves no
pixels on view cases; upstream-forced-to-copy is still 1.6×–4.0× faster
than the port. The port's disadvantage is real in both framings and is
kernel-latency (rows 1–4) plus the always-copy contract (rows 6–7,
9), not Python glue, at image sizes.

## Method (rerun for verification)

```bash
pixi run bench                       # flock-wrapped bench/bench.py, best of 15
# targeted measurements (best of 15–200, perf_counter; scratch in /tmp, not committed):
#  - split full call vs _lib._run vs raw lib.aics_gather on a reused plan
#  - ctypes-call count instrumented on lib.aics_gather (1 per call at 6.29M elems)
#  - cProfile on (2,3,64,64) uint16 and (2,3,1024,1024) uint8
#  - tracemalloc allocs per call; _contiguous_run / _selected_axes / _slice_indices micro-timings
#  - dtype sweep, axis-order sweep, 1D run-vs-gather, hot-vs-evicted, strided-view copy share
#  - upstream via tests.reference (tests/.refcache/4.14.0), incl. forced-copy comparison
pixi run python -c "import sys; sys.path.insert(0,'.'); from tests.reference import REFERENCE; print(REFERENCE.__file__)"
```

Notes / caveats: absolute times drift with host load (e.g. a best-of-19
re-run measured rows 1–4 ~1.5–1.7× slower while NumPy slowed
proportionally, ratios essentially unchanged) — treat absolute ms as
host- and moment-specific, and compare ratios/BW within a run. `bench.py`
prints shapes as `3x512x512` where the tuple is `(2,3,512,512)`; the
tuple is what was timed. Only `bench/analysis.md` was added; `src/`,
`tests/`, and `bench/bench.py` are unchanged.
