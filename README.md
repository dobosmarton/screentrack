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

Share of IDs that still named the same object two seconds later, from the same benchmark. The recorded clips come from three matches, with unit tracks reviewed by eye:

| Footage | screentrack | Simple IoU matcher |
|---|---|---|
| Generated scenes | 86.9% | 88.8% |
| Recorded game, 6 clips | 78.8% | 94.1% |

On the recorded clips, screentrack's ID names a different object 0.24% of the time, against 0.07% for IoU. A second label on the same spot doesn't count as a different object, because a click lands in the same place.

So screentrack is fast but doesn't yet keep identities better than plain IoU matching. The AoE2 tracker it was ported from has since changed, in [aoe2-agent#11](https://github.com/dobosmarton/aoe2-agent/pull/11):

- Matches rank by overlap.
- A label change needs the same box.
- Static objects are remembered for 5 s.

On the clips from the second and third matches, those rules keep 96.5% of IDs against 94.4% for IoU. However, they were developed on those clips, so they'll be ported only after they pass a test on a newly recorded match.

## Parity

screentrack makes the same identity decisions as the AoE2 tracker at the reference commit named in `fixtures/manifest.json`. [docs/SPEC.md](docs/SPEC.md) defines the contract, and `fixtures/` holds golden output from that tracker, generated on Python 3.11. Don't edit fixtures by hand.

Clippy enforces the numeric rules in `clippy.toml`. For example, `f64::hypot` is banned because it disagrees with Python's `math.hypot` on about one input in ten.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. See [NOTICE](NOTICE) for third-party attribution.
