# screentrack

A Rust library with Python bindings that gives stable identities to objects detected in screen captures. A detector returns independent boxes for each frame; screentrack links them across frames, so each object keeps one ID and a smoothed label.

It was extracted from the [AoE2 agent](https://github.com/dobosmarton/aoe2-agent)'s temporal tracker, which now uses it by default. Tuned values live in `Config`, and the core has no game-specific types.

## How it works

For each frame, you pass the detector's boxes, labels and confidences, an appearance signature per box, and how the camera moved. screentrack then:

1. **Expires old tracks.** A unit is forgotten after 1 s without a box, and a static object, such as a building or a resource, after 5 s. Both stretch to three frame gaps when frames are slow.
2. **Pairs tracks with boxes.** A pair must be within reach of the track's speed and look similar. A new label needs the boxes to overlap by half, and a static object needs a tenth. Cheaper pairs overlap more, look more alike and keep their label.
3. **Refuses to guess.** When a track or box has two near-equal options, neither is matched, and an ID never falls back to its second choice. A new ID is safer than the wrong one.
4. **Smooths labels** with a confidence-weighted, time-decayed vote over recent observations.

You name the labels of objects that move. Every other label is treated as static. [docs/SPEC.md](docs/SPEC.md) defines every rule exactly.

## Installation

Python 3.11 or newer. Wheels for macOS on Apple Silicon, Linux x86_64 and Windows x64 are attached to each [GitHub release](https://github.com/dobosmarton/screentrack/releases). screentrack is not on PyPI, so install a wheel by URL:

```sh
uv pip install https://github.com/dobosmarton/screentrack/releases/download/v0.1.0/screentrack-0.1.0-cp311-abi3-macosx_11_0_arm64.whl
```

The other wheels end in `-manylinux_2_28_x86_64.whl` and `-win_amd64.whl`. To depend on it from a uv project, list one URL per platform under `[tool.uv.sources]`, as the AoE2 agent's root `pyproject.toml` does.

For Rust, the crate isn't on crates.io either:

```toml
[dependencies]
screentrack = { git = "https://github.com/dobosmarton/screentrack", tag = "v0.1.0" }
```

## Usage

### Python

```python
import numpy as np
import screentrack

# Labels not listed here are static objects: they're remembered longer and only
# continue on a box that overlaps their last one.
tracker = screentrack.Tracker(screentrack.Config.aoe2(), unit_labels={"villager", "sheep"})

pixels = np.zeros((360, 640, 3), dtype=np.uint8)  # one RGB frame, (H, W, 3), C-contiguous
boxes = np.array([[100, 100, 120, 140], [300, 100, 360, 160]], dtype=np.float64)
result = tracker.observe(
    capture_id=1,
    captured_at=1.0,  # seconds, from a monotonic clock
    motion="stable",  # or "reset", or ("pan", dx, dy) when the camera moved
    boxes=boxes,
    labels=["villager", "house"],
    confidences=np.array([0.9, 0.8]),
    signatures=screentrack.rgb_mean_std(pixels, boxes),
)
if result is not None:  # None means the frame was older than the last one
    track_ids, labels, agreements = result
```

- **`rgb_mean_std`** computes every box's signature in one call and releases the GIL while it runs, so other Python threads keep running.
- **`observe`** returns one entry per box, in order. An agreement is `NaN` until a track has two observations.
- **Diagnostics:** `ambiguous_count`, `candidate_count`, `candidate_ms` and `matching_ms` describe the last accepted frame.
- **Other methods:** `accepts(capture_id, captured_at)` tells whether a frame is new enough to be processed, and `reset()` drops every track but keeps the ID counter. IDs are never reused.

### Rust

```rust
use screentrack::{Config, Detection, Motion, Tracker, rgb_mean_std};

// Labels are integers you assign. Labels not listed are static objects.
const VILLAGER: u32 = 0;
const HOUSE: u32 = 1;
let mut tracker = Tracker::new(Config::aoe2(), [VILLAGER]);

let (width, height) = (640, 360);
let pixels = vec![0_u8; width * height * 3]; // RGB, row-major
let detect = |label, bbox| Detection {
    bbox,
    label,
    confidence: 0.9,
    signature: rgb_mean_std(&pixels, width, height, width * 3, bbox),
};
let frame = [
    detect(VILLAGER, [100.0, 100.0, 120.0, 140.0]),
    detect(HOUSE, [300.0, 100.0, 360.0, 160.0]),
];
if let Some(tracked) = tracker.observe(1, 1.0, Motion::Stable, &frame) {
    for item in &tracked {
        println!("track {} is {}", item.track_id, item.label);
    }
}
```

The `python` feature builds the bindings. The `trace` feature exposes intermediate decisions for parity tests.

## Performance

Total time per frame at steady state, p95 in milliseconds. It includes signatures and the conversions the agent pays for. The figures come from the AoE2 agent's `scripts/tracker_benchmark`, run on an Apple Silicon Mac with real annotated game frames and the real detector:

| Density band | Python | screentrack | Speedup | Speedup, lower bound |
|---|---|---|---|---|
| Low | 63.68 | 1.09 | 59x | 55x |
| Typical | 101.54 | 1.99 | 51x | 48x |
| High | 68.64 | 1.42 | 48x | 46x |
| Peak | 91.11 | 2.37 | 38x | 37x |

Signatures are most of the cost on real frames, because real boxes are far larger than the crowded scene in `benches/`. That scene and `just bench` catch regressions; they aren't headline numbers.

## Identity quality

The measure is the share of IDs that still named the same object two seconds later, the delay before an agent acts on what it saw. The baseline is a simple matcher that pairs boxes by overlap (IoU).

The rules were fixed before a game was recorded, then judged on it once. On that held-out recorded game, match 5:

| Check | AoE2 tracker | IoU matcher |
|---|---|---|
| Same object after 2 s | 95.8% | 93.0% |
| ID moved to a different object | 1 swap | 1 swap |
| Label accuracy | 99.61% | 99.61% (raw detector labels) |

The verdict was run on the Python tracker these rules come from. Given the same unit labels, screentrack's decisions match it frame for frame on that recording and on every other recorded clip: 2,987 frames and 96,810 tracked objects in all. Earlier versions of the rules failed matches 2 to 4. Each failure was studied, and each of those matches was then retired from testing. The [benchmark report](https://github.com/dobosmarton/aoe2-agent/blob/main/docs/tracker-benchmark.md) has the full history.

On the benchmark's generated scenes, screentrack keeps 89.0% of IDs against 88.8% for IoU.

## Parity

screentrack makes the same identity decisions as the Python tracker at the reference commit named in `fixtures/manifest.json`. [docs/SPEC.md](docs/SPEC.md) defines the contract, and `fixtures/` holds golden output from that tracker, generated on Python 3.11. Regenerate fixtures with `just fixtures`; don't edit them by hand.

Clippy enforces the numeric rules in `clippy.toml`. For example, `f64::hypot` is banned, because it disagrees with Python's `math.hypot` on about one input in ten.

## Development

You need [rustup](https://rustup.rs), [uv](https://docs.astral.sh/uv/), [just](https://just.systems) and [cargo-binstall](https://github.com/cargo-bins/cargo-binstall). The toolchain is pinned in `rust-toolchain.toml`. On Apple Silicon, make sure Rust builds natively:

```sh
rustup set default-host aarch64-apple-darwin
just setup   # installs pinned tools, the Python environment and git hooks
just check   # runs everything CI runs
```

| Command | What it does |
|---|---|
| `just dev` | Re-runs checks and tests on save |
| `just check` | Runs everything CI runs, with warnings as errors |
| `just test` | Runs the Rust and Python tests |
| `just py` | Builds the release module into the local venv |
| `just bench` | Runs the benchmarks |
| `just mutants` | Checks that the tests notice changes to the decision logic |
| `just coverage` | Reports test coverage |
| `just fixtures` | Regenerates fixtures from the AoE2 reference (set `AOE2_AGENT` to its checkout) |

Pushing a `v*` tag builds the wheels and attaches them to a GitHub release.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. See [NOTICE](NOTICE) for third-party attribution.
