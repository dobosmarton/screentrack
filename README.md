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

## Parity

screentrack must make the same identity decisions as the Python tracker it replaces. [docs/SPEC.md](docs/SPEC.md) defines the contract, and `fixtures/` holds golden output from that tracker, generated on Python 3.11. Don't edit fixtures by hand.

Clippy enforces the numeric rules in `clippy.toml`. For example, `f64::hypot` is banned because it disagrees with Python's `math.hypot` on about one input in ten.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. See [NOTICE](NOTICE) for third-party attribution.
