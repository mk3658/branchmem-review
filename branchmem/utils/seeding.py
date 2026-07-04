"""Central seeding for deterministic scenario generation and non-LLM components.

LLM API calls are NOT made deterministic by this — see llm/cache.py and
PROGRESS.md for the honesty requirement around LLM non-determinism.
"""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
