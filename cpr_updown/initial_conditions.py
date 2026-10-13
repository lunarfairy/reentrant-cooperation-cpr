"""Seeded initial-state generator; the runner explicitly selects Uniform(0, 1)."""
from __future__ import annotations
import numpy as np


INITIAL_LOWER = 0.05


INITIAL_UPPER = 0.95


DEFAULT_INITIAL_SEED = 82702


def generate_shared_initial_conditions(
    count: int = 50,
    seed: int = DEFAULT_INITIAL_SEED,
    lower: float = INITIAL_LOWER,
    upper: float = INITIAL_UPPER,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=(count, 4))
