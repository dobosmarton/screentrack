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
| Low | 60.60 | 0.98 | 62x | 60x |
| Typical | 94.93 | 1.71 | 56x | 54x |
| High | 66.74 | 1.48 | 45x | 40x |
| Peak | 86.95 | 2.48 | 35x | 33x |

Signature extraction is most of the cost on real frames, because real boxes are far larger than the crowded benchmark scene in `benches/`. That scene and `just bench` are for catching regressions, not for headline numbers.

## Identity quality

Share of IDs that still named the same object two seconds later, from the same benchmark:

| Footage | screentrack | Simple IoU matcher |
|---|---|---|
| Generated scenes | 86.9% | 88.8% |
| Recorded game, still camera | 80.1% | 94.2% |
| Recorded game, moving camera | 66.0% | 92.0% |

screentrack rarely points at the wrong object: 0.2% and 0.1% on the recorded clips, against 4.4% and 1.8% for IoU. Those wrong-target figures and the moving-camera row are rough, because automatic truth hasn't had its hand audit. The recorded rows come from one match.

Before a fix on 2026-11-01, survival on the recorded clips was 0.0% and 1.5%.

## Parity

screentrack must make the same identity decisions as the Python tracker it replaces. [docs/SPEC.md](docs/SPEC.md) defines the contract, and `fixtures/` holds golden output from that tracker, generated on Python 3.11. Don't edit fixtures by hand.

Clippy enforces the numeric rules in `clippy.toml`. For example, `f64::hypot` is banned because it disagrees with Python's `math.hypot` on about one input in ten.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. See [NOTICE](NOTICE) for third-party attribution.
