from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def deterministic_sample(
    values: list[T],
    *,
    max_cases: int | None,
    seed: int,
) -> list[T]:
    if max_cases is None or max_cases >= len(values):
        return values
    indices = sorted(random.Random(seed).sample(range(len(values)), max_cases))
    return [values[index] for index in indices]
