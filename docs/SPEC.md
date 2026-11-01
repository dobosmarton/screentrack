# screentrack — Specification and Implementation Plan

A Rust library (with Python bindings) that gives stable identities to objects detected in screen captures. It is extracted from the AoE2 LLM arena's `temporal_entities.py` and generalized only where the generalization is cheap: tuned constants become configuration, and game-specific types stay in an adapter inside the AoE2 repo.

> Status: accepted, 2026-10-31. Corrected after checking the draft against the reference code; see §11.

---

## 1. Goals and non-goals

### Goals

1. **Performance.** Bring temporal association well under the arena's 20 ms p95 perception budget. On real game frames at the detector's 5 Hz cadence, the Python tracker spends 64-95 ms p95 per frame at steady state, mostly on appearance extraction. Target: **≤ 2 ms p95 for about 60 objects at 5 Hz steady state, including appearance extraction**, on the dev Mac, measured with the AoE2 repo's `scripts/tracker_benchmark`. Measure separately on the Windows VM.
2. **Behavioural parity.** Make the same identity decisions as the current Python implementation on every recorded and synthetic sequence (see §6).
3. **A clean, generic core.** No game names, classes or types in the library. Every tuned value lives in a `Config` struct, and the AoE2 values ship as a named preset.
4. **Drop-in integration.** The AoE2 agent keeps its current API (`prepare_detections`, `TemporalEntities.observe`) through a thin adapter, behind a switch, so both implementations can run side by side.
5. **Learning.** Exercise core Rust skills on a bounded problem: ownership, slices, PyO3/maturin, zero-copy NumPy access, releasing the GIL, criterion benchmarks and cross-platform wheels.

### Non-goals (v1)

- A general multi-object tracking library competing with Trackforge or Similar (SORT, ByteTrack, Kalman filters, Re-ID models).
- Porting `camera_motion.py` (FFT phase correlation). It is designed for, but deferred to, phase 8.
- Velocity prediction, class-compatibility rules and per-region motion. The API leaves room for them; the implementation waits for a second real use case.
- Publishing to PyPI or crates.io.

---

## 2. Background: what the current tracker does

Source: `aoe2-llm-arena/agent/apps/agent/src/temporal_entities.py` (~280 lines).

The detector returns independent boxes per frame. The tracker links them across frames:

1. **Appearance.** For each box it computes a 6-number signature: the per-channel RGB mean and standard deviation of the crop, scaled to [0, 1].
2. **Ordering guard.** It ignores any result that is not newer than the last one by both `capture_id` and `captured_at`.
3. **Reset.** It drops all history on a camera reset, or when more than `history` seconds have passed since the last frame.
4. **Motion compensation.** It shifts surviving tracks by the camera pan `(dx, dy)` and drops tracks older than `history`.
5. **Candidate generation.** It keeps only plausible (track, detection) pairs, gated by distance-over-time and appearance difference. Horizontal pruning uses binary search over the sorted x-centres.
6. **Ambiguity rejection.** A track or detection whose two cheapest candidate costs are closer than `ambiguity_margin` is never matched.
7. **Greedy matching.** Remaining pairs are matched cheapest-first, one-to-one.
8. **Label vote.** Each track keeps its recent evidence (label, confidence, time). A confidence-weighted, time-decayed vote smooths label flicker.
9. **Output.** Each detection gets a stable track ID, a smoothed label, the raw label and an agreement score.

---

## 3. Functional specification

All arithmetic uses `f64` to match Python floats.

### 3.1 Configuration

| Field | AoE2 preset | Meaning |
|---|---|---|
| `history_s` | 1.0 | Track lifetime without observations. Also the frame-gap reset threshold and the evidence window. |
| `speed_px_per_s` | 65.0 | Allowed centre displacement per second of elapsed time. |
| `size_factor` | 0.75 | Allowed displacement as a fraction of the track's box size. |
| `min_size_px` | 12.0 | Floor for box size in the gate. |
| `max_appearance_distance` | 0.55 | Euclidean signature distance above which a pair is rejected. |
| `ambiguity_margin` | 0.15 | Cost difference below which two candidates count as a tie (strict `<`). |
| `stale_overlap` | 0.5 | Overlap above which an unmatched track is a stale duplicate of an updated one (strict `>`). |
| `max_observations` | 3 | Evidence entries kept per track. |
| `label_half_life_s` | 0.4 | Decay half-life for label votes. |
| `min_label_margin` | 0.25 | Required normalized margin to switch a label. |

`Config::aoe2()` returns exactly these values. `Default` is the same preset, and it is documented as such.

### 3.2 Inputs per frame

- `capture_id: i64`, `captured_at: f64` (seconds, monotonic clock).
- `motion: Motion`, one of `Stable`, `Pan { dx, dy }` (content displacement in full pixels) or `Reset`.
- `detections: &[Detection]`, where each `Detection` has:
  - `bbox: [f64; 4]` as `(x0, y0, x1, y1)`, with `x0 < x1` and `y0 < y1` guaranteed by the caller;
  - `label: LabelId` (an opaque `u32`; the bindings intern strings);
  - `confidence: f64`;
  - `signature: [f64; 6]`.

### 3.3 Appearance signature

`rgb_mean_std(pixels: H×W×3 u8 view, bbox) -> [f64; 6]`

1. Truncate each box coordinate toward zero, matching Python `int()`. Clamp every coordinate to `[0, W]` and `[0, H]`, so a box fully outside the frame is empty.
2. Return all zeros for an empty crop.
3. Otherwise compute per-channel `(sum / n) / 255` and population standard deviation (`ddof = 0`) `/ 255`, in the order `[mean_r, mean_g, mean_b, std_r, std_g, std_b]`. The mean uses two divisions, which reproduces NumPy exactly.

The signature is computed separately from `observe`, so callers can run it off the event loop. This mirrors the current `prepare_detections` split.

### 3.4 `observe` algorithm

Given tracks `T` (ordered), detections `D` (ordered) and time `t`:

1. **Guard.** If `capture_id <= last_capture_id` or `t <= last_at`, return `None` (Python returns an empty tuple) and change no state.
2. **Reset.** If `motion == Reset` or `t - last_at > history_s`, clear all tracks. The ID counter is **not** reset.
3. Set `last_capture_id` and `last_at`.
4. **Shift and expire.** Translate every track box by the pan shift (`(0, 0)` unless panning). Drop tracks with `t - last_seen_at > history_s`. Keep the order.
5. **Candidates.** For each track `i` (in order), with `elapsed = t - last_seen_at`:
   - `size = max(w, h, min_size_px)` of the track box;
   - `allowed = size_factor * size + speed_px_per_s * elapsed`;
   - `distance` = Euclidean distance between box centres, computed with a port of CPython's `math.hypot`; reject if `distance > allowed`;
   - `app` = Euclidean distance between signatures; reject if `app > max_appearance_distance`;
   - `cost = distance / allowed + app`.

   Pruning by x-centre (`old_x ± allowed`, inclusive bounds) is an optimization only, and it keeps Python's exact bounds and order. Sort candidates lexicographically by `(cost, track_index, detection_index)`.
6. **Ambiguity.** Walk the sorted list. The first two candidates seen for a track or detection are its cheapest. If `second - first < ambiguity_margin`, mark that track (or detection) ambiguous. Later candidates never count. Record `last_ambiguous_count` as the number of ambiguous detections.
7. **Matching.** Walk the sorted list again. Match `(i, j)` only if neither is ambiguous and neither is already used.
8. **Emit, in detection order.** For each detection `j`:
   - **If matched to track `i`:** reuse that track's ID and evidence. **Otherwise:** allocate a new ID from a monotonically increasing counter, with empty evidence.
   - Append `(label, confidence, t)` to the evidence. Keep entries with `t - captured_at <= history_s`, then keep the last `max_observations`.
   - Run the label vote (§3.5) to get `(label, agreement)`.
   - The output carries `track_id`, `label`, `raw_label`, `agreement` (`None` when fewer than 2 evidence entries) and the frame's metadata.
9. **New track list:** the tracks updated in step 8 (in detection order), followed by unmatched old tracks (in their previous order). An unmatched track is dropped if its intersection over union with any updated track exceeds `stale_overlap`. Otherwise ambiguous old tracks survive this way.

### 3.5 Label vote

Given evidence `E` (oldest first), with `now` = the time of the last entry, and the current raw label `c`:

1. If `|E| < 2`, return `(c, None)`.
2. Each label's score is the sum of `confidence * 2^(-(now - captured_at) / label_half_life_s)` over its entries. `total` is the plain left-to-right sum of all scores. If `total == 0`, return `(c, 0)`.
3. The winner is the label with the highest score. **Ties go to the label that first appears in `E`.** Python's `max()` over dict insertion order behaves this way, and parity depends on it.
4. `agreement = score(winner) / total`.
5. If `winner != c`, let `runner_up` be the best score among the other labels (0 if none) and `supporting` the number of entries labelled `winner`. If `supporting < 2` or `(score(winner) - runner_up) / total < min_label_margin`, return `(c, score(c) / total)`.
6. Otherwise return `(winner, agreement)`.

### 3.6 Diagnostics

The tracker exposes the last frame's `ambiguous_count`, `candidate_count`, `candidate_ms` and `matching_ms`. These replace the Python `last_*` attributes.

### 3.7 `reset()`

Clears tracks only. It does not touch the ID counter or the ordering guard, matching the current Python behaviour.

---

## 4. API design

### 4.1 Rust crate (pure, no Python)

```rust
pub struct Config { /* fields from §3.1 */ }
impl Config { pub fn aoe2() -> Self; }

pub type LabelId = u32;
pub type Signature = [f64; 6];

pub enum Motion { Stable, Pan { dx: f64, dy: f64 }, Reset }

pub struct Detection { pub bbox: [f64; 4], pub label: LabelId, pub confidence: f64, pub signature: Signature }

pub struct Tracked { pub track_id: u64, pub label: LabelId, pub raw_label: LabelId, pub agreement: Option<f64> }

pub struct Tracker { /* tracks, next_id, last_capture_id, last_at, config, stats */ }
impl Tracker {
    pub fn new(config: Config) -> Self;
    pub fn accepts(&self, capture_id: i64, captured_at: f64) -> bool;
    pub fn observe(&mut self, capture_id: i64, captured_at: f64, motion: Motion, detections: &[Detection]) -> Option<Vec<Tracked>>;
    pub fn reset(&mut self);
    pub fn stats(&self) -> &FrameStats;
}

pub fn rgb_mean_std(pixels: &[u8], width: usize, height: usize, row_stride: usize, bbox: [f64; 4]) -> Signature;
```

Design choices:

- **Labels are integers in the core.** String handling stays at the boundary, which keeps evidence storage compact and comparisons cheap.
- **Signature is a fixed `[f64; 6]` in v1.** Phase 8 can generalize it to a trait (`fn signature(crop) -> [f32; N]`). Don't build that before a second signature exists.
- **No `numpy` or `ndarray` dependency in the core.** The pixel view is a raw slice plus dimensions and stride.

### 4.2 Python bindings (PyO3, behind a `python` cargo feature)

Generic, array-in and array-out. They know nothing about `EntitySnapshot`.

```python
import screentrack

sigs = screentrack.rgb_mean_std(pixels: np.ndarray[H, W, 3, uint8], boxes: np.ndarray[N, 4, float64]) -> np.ndarray[N, 6]

tracker = screentrack.Tracker(screentrack.Config.aoe2())
result = tracker.observe(
    capture_id=..., captured_at=...,
    motion=("pan", dx, dy) | "stable" | "reset",
    boxes=..., labels=list[str], confidences=..., signatures=...,
)  # -> None or (track_ids: ndarray[N, uint64], labels: list[str], agreements: ndarray[N] with NaN for None)
```

- `rgb_mean_std` borrows the NumPy buffer without copying, processes all boxes in one call, and **releases the GIL** while computing.
- Label interning lives in the binding layer, with one table per `Tracker`.
- Type stubs (`.pyi`) ship with the package, so basedpyright works in the AoE2 repo.

### 4.3 AoE2 adapter (lives in the AoE2 repo, not the library)

`temporal_entities.py` keeps its public names (`PreparedDetection`, `prepare_detections`, `TemporalEntities`) and turns into a thin translator:

- **`prepare_detections`:** collects boxes, calls `screentrack.rgb_mean_std` once and wraps the results.
- **`TemporalEntities.observe`:** converts `CameraMotion` to the motion tuple, calls the tracker and rebuilds `EntitySnapshot`s with `track_{id}`, labels, agreement and frame metadata.
- **Backend switch:** an env var or config flag selects `native`, `python` or `shadow`. `shadow` runs both and logs any difference.
- The current Python implementation moves to `temporal_entities_py.py` as the reference. It stays until the native path has run cleanly in real games.

---

## 5. Repository layout

```
screentrack/
  Cargo.toml              # crate; features: python (pyo3, numpy)
  pyproject.toml          # maturin backend, python-source = "python"
  src/
    lib.rs                # public API, re-exports
    config.rs
    signature.rs          # rgb_mean_std
    candidates.rs         # gating, cost, pruning, sort
    matching.rs           # ambiguity + greedy
    label.rs              # vote
    tracker.rs            # Tracker state machine
    python.rs             # #[cfg(feature = "python")] bindings
  python/screentrack/
    __init__.py
    __init__.pyi
  tests/                  # Rust integration tests (fixture replay)
  fixtures/               # JSON sequences exported from the Python reference
  benches/                # criterion benchmarks
  pytests/                # Python parity + binding tests
  .github/workflows/      # CI + wheels
  README.md
```

Single crate with a feature flag rather than a workspace: simpler while there is only one consumer.

Toolchain:

- Rust stable, with `clippy -D warnings` and `rustfmt`.
- Python ≥ 3.11, built as an **abi3-py311** wheel, so one wheel per platform covers all Python versions.
- maturin, uv for the Python side, pytest.

---

## 6. Parity and testing strategy

### 6.1 Parity definition

- **Must be identical:** candidate sets, ambiguity sets, matches, track IDs, output labels, `None` versus present agreement, and track list order.
- **May differ by ≤ 1e-9 (absolute):** signature values, costs and agreement scores.
- **Numeric sources of difference:** `u8` sums are exact in `f64`, so the mean matches NumPy bit for bit. The standard deviation uses an exact integer formula and stays within 2e-13 of NumPy. Python's `x ** 2` goes through libm `pow` and differs from `x * x` on about 0.14% of inputs, so costs are not guaranteed bit-exact, which is why the tolerance exists. Tracker replay uses the signatures stored in the fixtures, so signature tolerance cannot hide a tracker difference.

### 6.2 Test layers

1. **Rust unit tests.** Port every case from `test_temporal_association_scaling.py` and the tracker cases in `test_stream_perception.py`:
   - ambiguity at exactly 0.15 is not a tie;
   - equal costs are a tie;
   - close costs below a clear winner do not count;
   - plus reset, pan, expiry, guard and label-vote edge cases.
2. **Exhaustive oracle.** A brute-force `O(n·m)` candidate and ambiguity implementation in Rust test code. Property tests (`proptest`) check that the pruned version matches it on random scenes, mirroring the existing Python oracle test.
3. **Golden fixtures.** A small exporter script in the AoE2 repo runs the *Python* tracker over:
   - the synthetic sequences from the existing tests;
   - crowded 20/40/60-object scenes;
   - recorded sequences from `sequence_replay`.

   It writes inputs and outputs as JSON. Rust integration tests replay them.
4. **Python parity tests.** Run both backends over the same inputs and compare per §6.1. Run the AoE2 repo's existing tests against the native backend.
5. **Shadow mode in real games** (phase 6): log every divergence with the frame ID.

---

## 7. Performance plan

- **Baseline first.** Record Python p50/p95 with the AoE2 repo's `scripts/bench_temporal.py`, before any Rust exists. Do this on both the Mac and the Windows VM. Measure at steady state: ambiguous tracks survive for `history_s`, so a crowded scene keeps slowing down for about a second.
- **Rust micro-benchmarks** with criterion: `rgb_mean_std` on 60 boxes, `observe` on crowded scenes, and the label vote.
- **End-to-end Python benchmark** through the adapter, because conversion overhead counts. If rebuilding `EntitySnapshot`s dominates, consider returning plain tuples.
- **Optimizations, only if needed:**
  - store detections as struct-of-arrays;
  - reuse candidate buffers across frames (no per-frame allocation);
  - make `rgb_mean_std` cache-friendly by walking rows with the stride;
  - optionally use `rayon` over boxes. This is probably not worth it at 60 boxes, so measure first.
- **Acceptance:** crowded-60 association plus signatures at ≤ 2 ms p95 at 5 Hz steady state on the Mac, and a measured, recorded figure on the Windows VM.

---

## 8. Implementation plan

Each phase ends with something runnable and a clear "done" check.

### Phase 0: Baseline and fixtures (in the AoE2 repo)
- Record current Python timings (Mac and VM).
- Write the fixture exporter and generate the initial fixture set.
- **Done when:** fixtures are committed and the baseline numbers are written down.

### Phase 1: Repository skeleton
- `cargo new --lib`, maturin `pyproject.toml`, the `python` feature, README, CI running `fmt`, `clippy` and `test` on macOS.
- **Done when:** `maturin develop` builds an empty module that imports from Python.

### Phase 2: Pure Rust core
- `config`, `signature`, `candidates`, `matching`, `label`, `tracker`, in that order, each with unit tests.
- **Done when:** all ported unit tests pass.

### Phase 3: Oracle and fixture replay
- Exhaustive oracle, proptest, and the fixture replay tests.
- **Done when:** every fixture replays identically per §6.1.

### Phase 4: Python bindings
- `rgb_mean_std` with zero-copy NumPy access and the GIL released; the `Tracker` class; label interning; `.pyi` stubs.
- **Done when:** Python parity tests pass against the fixtures.

### Phase 5: Benchmarks
- criterion benchmarks and the end-to-end Python benchmark.
- Optimize only where measurements point.
- **Done when:** the §7 acceptance targets are met on the Mac, or the reason they aren't is understood.

### Phase 6: AoE2 integration
- Add the path dependency through `[tool.uv.sources]`.
- Move the reference implementation to `temporal_entities_py.py`; turn `temporal_entities.py` into the adapter with the backend switch.
- Run the AoE2 test suite on both backends, then shadow mode in real games.
- **Done when:** the full test suite is green on `native` and shadow mode shows no divergences over several games.

### Phase 7: Windows x64 wheel
- GitHub Actions `maturin-action` matrix: `macos-arm64` and `windows-x64` (abi3).
- Install on the Windows ARM64 VM under x64 Python. Rerun the benchmarks there.
- **Done when:** the agent runs a real game on the VM with the `native` backend, and the VM timings are recorded.

### Phase 8: Later, each only when a real need appears
- Port `camera_motion.py` (`rustfft`).
- Signature trait for pluggable appearance signatures.
- Optional class-compatibility rules.
- Constant-velocity prediction.
- Per-region motion, for GUIs where only one pane scrolls.
- Remove the Python reference once the native path has been stable for a while.
- Compare against Trackforge's ByteTrack on the same fixtures, to document why a custom tracker is warranted.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Floating-point edge ties flip decisions | Parity failures that look like bugs | Exact integer sums for signatures; investigate each divergence; document true ties (§6.1) |
| Adapter conversion overhead eats the gain | Smaller end-to-end win than micro-benchmarks show | Measure end-to-end; batch calls; return arrays or tuples instead of objects if needed |
| Windows x64 wheel build friction | Can't deploy to the game VM | Set up CI wheels early (move to right after phase 1 if it's painful); abi3 keeps the matrix to one wheel per OS |
| Emulated x64 on ARM64 behaves differently | Different timings on the VM | Benchmark on the VM explicitly; don't assume the Mac numbers carry over |
| Scope creep into a general tracker | The project stalls | Non-goals in §1; phase 8 items only with a concrete second use case |
| Python reference drifts while porting | Parity checked against a moving target | Freeze the reference at a tagged commit; regenerate fixtures only deliberately |

---

## 10. Decisions

1. **Name.** `screentrack`.
2. **Repository.** Public, at `dobosmarton/screentrack`, checked out beside the agent at `aoe2-llm-arena/screentrack`.
3. **Integration mode.** Stay out of the agent's `uv.lock` until wheels exist, because its CI runs on Linux without this repository. Use `uv run --with-editable` during development, then per-platform wheel URLs. A Linux x64 wheel joins the macOS arm64 and Windows x64 wheels.
4. **Parity bar.** Identical decisions, with a 1e-9 tolerance on scores and signatures.
5. **Output shape.** Rebuild `EntitySnapshot`s first, and change that only if measurements show it dominates.

---

## 11. Corrections to the draft

Checked against the reference on Python 3.11.9 and NumPy 2.4.6.

- **Baseline.** The 15–21 ms figure was preparation plus association for frames 2 and 3 only. Steady state is about twice that at 5 Hz.
- **Mean.** Pairwise summation cannot change a `u8` sum. `(sum / n) / 255` matched NumPy on 9000 of 9000 channels; a single division matched on 7039.
- **`hypot`.** `f64::hypot` differed from `math.hypot` on 10.7% of inputs, and `sqrt(x² + y²)` on 14.2%. A scalar port of `vector_norm` differed on none of 1,000,000.
- **`sum()`.** Python's float `sum()` is compensated from 3.12, so fixtures must come from Python 3.11, and Rust uses plain sums.
- **Reference bugs.** A negative box end wrapped through a Python slice, and an all-zero-confidence vote could raise `ZeroDivisionError`. Both were fixed in the reference (AoE2 commit `4c61753`) to match this spec, and the fixtures come from the fixed version.
- **Benchmark scene.** The crowded synthetic scene used 64×64 boxes reading 0.25M pixels per frame. Real frames read 2.3M at the median and up to 8.2M, so the Python cost above comes from real frames instead. 219 unique annotated frames were available.
- **Reference ambiguity rule.** The original rule rejected an identity when any two of its candidates were close, and left stale duplicates behind. On recorded footage that rejected 97% of detections, so both changed in the reference (AoE2 commit `7d2244e`) and here. See §3.4 and the AoE2 repo's `docs/tracker-benchmark.md`.
- **Recorded sequences.** One match is recorded so far, from the Mac at 15 fps. See the AoE2 repo's `scripts/tracker_benchmark/recorded.py`. A VM capture is still a gate for shadow mode.
