# screentrack

A Rust library with Python bindings that gives stable identities to objects detected in screen captures. A detector returns independent boxes for each frame; screentrack links them across frames, so each object keeps one ID and a smoothed label.

It was extracted from the AoE2 LLM arena's temporal tracker. Tuned values live in `Config`, and the core has no game-specific types.

## Getting started

You need [rustup](https://rustup.rs), [uv](https://docs.astral.sh/uv/), [just](https://just.systems) and [cargo-binstall](https://github.com/cargo-bins/cargo-binstall). On Apple Silicon, make sure Rust builds natively:

```sh
rustup set default-host aarch64-apple-darwin
just setup   # installs pinned tools, the Python environment and git hooks
just check   # runs everything CI runs
```

## Commands

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

## Performance

Total time per frame at steady state, p95 in milliseconds, including signatures and the conversions the agent pays for. This comes from the AoE2 repo's `scripts/tracker_benchmark`, which uses real annotated game frames and the real detector on an Apple Silicon Mac:

| Density band | Python | screentrack | Speedup | Speedup, lower bound |
|---|---|---|---|---|
| Low | 64.14 | 1.55 | 43x | 35x |
| Typical | 95.01 | 1.89 | 50x | 49x |
| High | 70.68 | 1.86 | 38x | 34x |
| Peak | 92.94 | 3.11 | 30x | 25x |

Signature extraction is most of the cost on real frames, because real boxes are far larger than the crowded benchmark scene in `benches/`. That scene and `just bench` are for catching regressions, not for headline numbers.

## Known issue

screentrack deliberately makes the same decisions as the Python tracker it replaces, including its weakness: a stationary object can lose its identity permanently after one duplicate detection. In the same benchmark, only 18.8% of IDs still named the same object two seconds later, against 88.8% for a simple IoU matcher. Speed is solved; identity quality is not.

## Parity

screentrack must make the same identity decisions as the Python tracker it replaces. [docs/SPEC.md](docs/SPEC.md) defines the contract, and `fixtures/` holds golden output from that tracker, generated on Python 3.11. Don't edit fixtures by hand.

Clippy enforces the numeric rules in `clippy.toml`. For example, `f64::hypot` is banned because it disagrees with Python's `math.hypot` on about one input in ten.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. See [NOTICE](NOTICE) for third-party attribution.
