# screentrack commands. `just check` runs everything CI runs.

set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["bash", "-euo", "pipefail", "-c"]

aoe2 := env("AOE2_AGENT", "../agent")
decision_modules := "--file src/candidates.rs --file src/matching.rs --file src/label.rs"

# Show available commands.
default:
    @just --list

# Install the pinned tools, the Python environment and the git hooks.
setup:
    cargo binstall --no-confirm --locked \
        bacon@3.26.0 \
        cargo-deny@0.20.2 \
        cargo-hack@0.6.45 \
        cargo-llvm-cov@0.9.1 \
        cargo-machete@0.9.2 \
        cargo-mutants@27.1.0 \
        cargo-nextest@0.9.150 \
        samply@0.13.1
    uv sync --locked
    uv run --no-sync pre-commit install --hook-type pre-commit --hook-type pre-push

# Re-run checks and tests on save.
dev:
    bacon nextest -- --features trace

# Run everything CI runs, with warnings as errors.
check:
    CARGO_BUILD_WARNINGS=deny just lint deps features docs test

# Run formatters, static checks and Clippy.
lint:
    uv run --locked pre-commit run --all-files
    cargo clippy --locked --all-targets --all-features

# Check advisories, licenses, sources and unused dependencies.
deps:
    cargo deny --locked check
    cargo machete

# Confirm each cargo feature compiles on its own.
features:
    cargo hack --locked check --each-feature --no-dev-deps

# Build the API docs, failing on any rustdoc warning.
docs:
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

# Run the Rust and Python tests.
test: test-rust test-python

# Run Rust tests. `python` is left out so pure tests never link libpython.
test-rust:
    cargo nextest run --locked --features trace --no-tests pass
    cargo test --locked --doc --features trace

# Run Python tests, type checks and the stub check.
test-python:
    uv run --locked pytest
    uv run --locked basedpyright
    uv run --locked python -m mypy.stubtest screentrack

# Build the release module into the local venv.
py:
    uv run --no-sync maturin develop --release --uv

# Run the benchmarks against release builds.
bench:
    cargo bench --locked --features trace

# Check that the tests notice changes to the decision logic.
mutants:
    cargo mutants --locked --features trace --test-tool nextest {{ decision_modules }}

# Report test coverage.
coverage:
    cargo llvm-cov nextest --locked --features trace --summary-only

# Regenerate fixtures from the frozen AoE2 reference. Set AOE2_AGENT to its checkout.
fixtures:
    cd "{{ aoe2 }}" && uv run python scripts/export_screentrack_fixtures.py --output "{{ justfile_directory() }}/fixtures"
