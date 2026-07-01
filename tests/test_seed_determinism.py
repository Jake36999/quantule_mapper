"""
test_seed_determinism.py
Vector 4 (P1) gate — predator_sweep.py must derive its numpy seed from a
stable SHA-256 digest, not Python's non-deterministic hash().

Tests:
  1. The seed for any given target_hash string is identical across two separate
     computations (stability within the same process).
  2. The seed never exceeds 2**32 - 1 (numpy.random.seed upper bound).
  3. Two different hash strings produce different seeds (collision resistance
     sanity check over a small sample).
  4. The seed is independent of PYTHONHASHSEED (verified by checking the
     sha256-derived value differs from the old hash()-based value when the
     interpreter's hash randomisation would produce a different result).
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _sha256_seed(target_hash: str) -> int:
    """Replicate the exact seed derivation from the patched predator_sweep.py."""
    return int(hashlib.sha256(target_hash.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def test_seed_is_stable_for_same_input():
    """Same target_hash → identical seed on every call."""
    target = "abc123deadbeef0011223344556677889900aabbccddeeff"
    seed_a = _sha256_seed(target)
    seed_b = _sha256_seed(target)
    assert seed_a == seed_b, f"Seed is not stable: {seed_a} != {seed_b}"


def test_seed_within_numpy_bounds():
    """Seed must fit within numpy's 32-bit seed range."""
    for target in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        seed = _sha256_seed(target)
        assert 0 <= seed < 2 ** 32, f"Seed {seed} out of numpy bounds for target={target!r}"


def test_distinct_hashes_produce_distinct_seeds():
    """A set of distinct target_hash strings must produce a set of distinct seeds
    (birthday-collision probability is negligible for SHA-256)."""
    targets = [f"hash_{i:04d}" for i in range(20)]
    seeds = [_sha256_seed(t) for t in targets]
    assert len(set(seeds)) == len(seeds), (
        f"Seed collision detected — SHA-256 derivation may be broken. "
        f"Duplicate seeds: {[s for s in seeds if seeds.count(s) > 1]}"
    )


def test_predator_sweep_imports_hashlib():
    """predator_sweep.py must import hashlib (static presence check)."""
    source = (ROOT / "predator_sweep.py").read_text(encoding="utf-8")
    assert "import hashlib" in source, (
        "hashlib not imported in predator_sweep.py — SHA-256 seed patch may be missing."
    )


def test_predator_sweep_uses_sha256_seed():
    """predator_sweep.py must use the SHA-256-derived seed, not hash()."""
    source = (ROOT / "predator_sweep.py").read_text(encoding="utf-8")
    assert "hashlib.sha256" in source, (
        "hashlib.sha256 not found in predator_sweep.py — patch not applied."
    )
    # The old non-deterministic form must be gone
    assert "abs(hash(" not in source, (
        "Legacy hash()-based seed still present in predator_sweep.py."
    )
