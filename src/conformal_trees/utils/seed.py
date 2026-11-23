"""Seed handling. Every random operation in the project must go through here."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (for code we do not control)."""
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Preferred way to get randomness inside our own code."""
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *keys: object) -> int:
    """Deterministic 32-bit sub-seed from a base seed and arbitrary keys.

    Example: derive_seed(3, "split", "test"). Different components get
    independent-looking seeds, but the whole run is still reproducible from
    the single base seed. The result is a valid sklearn ``random_state``.
    """
    payload = "|".join(str(k) for k in (base_seed, *keys)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
