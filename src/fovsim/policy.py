"""Binary Base/Enhancement-3 policy."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    threshold: float = 0.5
    base_level: int = 0
    enhancement_level: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and within [0, 1]")
        if self.base_level != 0 or self.enhancement_level != 3:
            raise ValueError("The supported policy levels are fixed to Base=0 and E3=3")


def classify_fraction(
    contributing_gaussian_fraction: float,
    config: PolicyConfig,
) -> tuple[bool, int]:
    if not math.isfinite(contributing_gaussian_fraction):
        raise ValueError("contributing_gaussian_fraction must be finite")
    if not 0.0 <= contributing_gaussian_fraction <= 1.0:
        raise ValueError(
            "contributing_gaussian_fraction must be within [0, 1]"
        )
    use_enhancement = contributing_gaussian_fraction >= config.threshold
    return (
        use_enhancement,
        config.enhancement_level if use_enhancement else config.base_level,
    )
