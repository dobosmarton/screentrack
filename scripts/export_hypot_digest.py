"""Fingerprint `math.hypot` over a million inputs, for the Rust port to reproduce.

The fixture cases store each value, which limits how many fit. A fingerprint covers far
more inputs in a few bytes, which is what exposes rounding differences this rare.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

CASES = 1_000_000
SEED = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64_MASK = (1 << 64) - 1
OUTPUT = Path(__file__).parent.parent / "fixtures" / "hypot_digest.json"


def next_state(state: int) -> int:
    state ^= (state << 13) & U64_MASK
    state ^= state >> 7
    return state ^ ((state << 17) & U64_MASK)


def coordinate(state: int) -> float:
    return (state >> 11) / 2.0**53 * 1000.0 - 500.0


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("run this on Python 3.11, the version the fixtures come from")
    state = SEED
    digest = FNV_OFFSET
    for _ in range(CASES):
        state = next_state(state)
        x = coordinate(state)
        state = next_state(state)
        y = coordinate(state)
        bits = struct.unpack("<Q", struct.pack("<d", math.hypot(x, y)))[0]
        digest = ((digest ^ bits) * FNV_PRIME) & U64_MASK
    record = {"cases": CASES, "seed": SEED, "digest": f"{digest:016x}"}
    OUTPUT.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
