"""
MCEI — Macro Conditions & Expectations Index

Computes a composite macro/liquidity index from FRED series data.

Pipeline:
1. Raw FRED series → 2. Transform (YoY, spread, level) → 3. Percentile/Z-score normalization
4. Sign alignment (higher = expansionary) → 5. Weighted composite → 6. Regime classification

Weights are initial estimates. Regime thresholds must be validated with
historical walk-forward testing rather than treated as permanent truths.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from marketmaster.config.mcei_series import (
    MCEI_COMPONENTS,
    MCEI_REGIME_THRESHOLDS,
    MCEIComponent,
)


@dataclass
class ComponentResult:
    """Result of computing a single MCEI component."""
    name: str
    raw_value: float
    percentile: float          # 0-100
    z_score: float
    sign: str
    weight: float
    normalized: float           # sign-aligned 0-100 (higher = expansionary)
    category: str
    description: str


@dataclass
class MCEIResult:
    """Result of computing the full MCEI composite."""
    score: float                           # 0-100 composite
    regime: str                            # regime classification
    liquidity_regime: str                  # MCEI-specific regime
    components: dict[str, ComponentResult] # per-component breakdown
    weights_version: str
    computed_at: date
    raw_components: dict[str, float]        # raw values for storage

    def to_dict(self) -> dict:
        """Serialize for JSONB storage in mcei_history table."""
        return {
            "score": self.score,
            "regime": self.regime,
            "components": {
                name: {
                    "raw_value": c.raw_value,
                    "percentile": c.percentile,
                    "z_score": c.z_score,
                    "normalized": c.normalized,
                    "weight": c.weight,
                    "category": c.category,
                    "sign": c.sign,
                }
                for name, c in self.components.items()
            },
            "weights_version": self.weights_version,
        }


def _to_percentile(value: float, history: list[float]) -> float:
    """
    Convert a raw value to its percentile rank in the historical distribution.

    Uses linear interpolation between closest ranks.
    """
    if not history:
        return 50.0
    sorted_hist = sorted(history)
    n = len(sorted_hist)
    if n == 1:
        return 50.0 if value >= sorted_hist[0] else 0.0

    # Find position using binary search
    lo = 0
    for i, v in enumerate(sorted_hist):
        if value <= v:
            lo = i
            break
        lo = n
    if lo == 0:
        return 0.0
    if lo == n:
        return 100.0

    # Linear interpolation
    lower = sorted_hist[lo - 1]
    upper = sorted_hist[lo]
    if upper == lower:
        return (lo - 0.5) / n * 100
    frac = (value - lower) / (upper - lower)
    return ((lo - 1) + frac) / n * 100


def _to_zscore(value: float, history: list[float]) -> float:
    """Convert a raw value to a z-score relative to historical distribution."""
    if not history or len(history) < 2:
        return 0.0
    mean = np.mean(history)
    std = np.std(history, ddof=1)
    if std == 0:
        return 0.0
    return (value - mean) / std


def _align_sign(percentile: float, sign: str) -> float:
    """
    Align sign so that HIGHER always means more expansionary.

    For 'pos' components: higher raw value = higher percentile = expansionary → keep as is
    For 'neg' components: higher raw value = contractionary → flip to (100 - percentile)
    """
    if sign == "neg":
        return 100.0 - percentile
    return percentile


def _classify_regime(score: float) -> str:
    """Classify MCEI score into a regime."""
    if score >= MCEI_REGIME_THRESHOLDS["STRONG_EXPANSION"]:
        return "STRONG_EXPANSION"
    elif score >= MCEI_REGIME_THRESHOLDS["EXPANSION"]:
        return "EXPANSION"
    elif score >= MCEI_REGIME_THRESHOLDS["NEUTRAL"]:
        return "NEUTRAL"
    elif score >= MCEI_REGIME_THRESHOLDS["CONTRACTION"]:
        return "CONTRACTION"
    else:
        return "STRONG_CONTRACTION"


def compute_component(
    component: MCEIComponent,
    raw_value: float,
    history: list[float],
) -> ComponentResult:
    """
    Compute a single MCEI component from raw value and historical distribution.

    Args:
        component: MCEI component configuration
        raw_value: Current raw value of the series
        history: Historical values for percentile/z-score computation

    Returns:
        ComponentResult with all computed values
    """
    percentile = _to_percentile(raw_value, history)
    z_score = _to_zscore(raw_value, history)
    normalized = _align_sign(percentile, component.sign)

    return ComponentResult(
        name=component.name,
        raw_value=raw_value,
        percentile=round(percentile, 4),
        z_score=round(z_score, 4),
        sign=component.sign,
        weight=component.weight,
        normalized=round(normalized, 4),
        category=component.category,
        description=component.description,
    )


def calculate_mcei(
    component_values: dict[str, float],
    component_histories: dict[str, list[float]],
    weights_version: str = "v1",
    as_of_date: Optional[date] = None,
) -> MCEIResult:
    """
    Calculate the MCEI composite score from component values and histories.

    Args:
        component_values: {component_name: raw_value} for each available component
        component_histories: {component_name: [historical values]} for percentile computation
        weights_version: Version tag for the weight set used
        as_of_date: Date of the computation (defaults to today)

    Returns:
        MCEIResult with composite score, regime, and per-component breakdown
    """
    if not component_values:
        return MCEIResult(
            score=50.0,
            regime="NEUTRAL",
            liquidity_regime="NEUTRAL",
            components={},
            weights_version=weights_version,
            computed_at=as_of_date or date.today(),
            raw_components={},
        )

    components: dict[str, ComponentResult] = {}
    total_weight = 0.0
    weighted_sum = 0.0
    raw_components: dict[str, float] = {}

    for comp_config in MCEI_COMPONENTS:
        if comp_config.name not in component_values:
            continue

        raw_value = component_values[comp_config.name]
        history = component_histories.get(comp_config.name, [])

        result = compute_component(comp_config, raw_value, history)
        components[comp_config.name] = result
        raw_components[comp_config.name] = raw_value

        weighted_sum += result.normalized * comp_config.weight
        total_weight += comp_config.weight

    if total_weight == 0:
        score = 50.0
    else:
        score = weighted_sum / total_weight

    score = max(0.0, min(100.0, round(score, 2)))
    regime = _classify_regime(score)

    return MCEIResult(
        score=score,
        regime=regime,
        liquidity_regime=regime,
        components=components,
        weights_version=weights_version,
        computed_at=as_of_date or date.today(),
        raw_components=raw_components,
    )


# Backward compatibility with the original stub interface
def calculate_mcei_legacy(
    components: dict[str, float],
    weights: dict[str, float],
) -> MCEIResult:
    """Legacy interface for backward compatibility with the v1 stub."""
    if not components:
        return MCEIResult(
            score=50.0, regime="NEUTRAL", liquidity_regime="NEUTRAL",
            components={}, weights_version="legacy",
            computed_at=date.today(), raw_components={},
        )

    total_weight = sum(weights.get(k, 0.0) for k in components)
    if total_weight <= 0:
        raise ValueError("MCEI weights must contain positive weight")

    score = sum(components[k] * weights.get(k, 0.0) for k in components) / total_weight
    score = max(0.0, min(100.0, score))

    if score >= 80:
        regime = "STRONG_EXPANSION"
    elif score >= 60:
        regime = "EXPANSION"
    elif score >= 40:
        regime = "NEUTRAL"
    elif score >= 20:
        regime = "CONTRACTION"
    else:
        regime = "STRONG_CONTRACTION"

    return MCEIResult(
        score=score, regime=regime, liquidity_regime=regime,
        components={k: ComponentResult(
            name=k, raw_value=v, percentile=v, z_score=0.0,
            sign="pos", weight=weights.get(k, 0.0), normalized=v,
            category="legacy", description="",
        ) for k, v in components.items()},
        weights_version="legacy",
        computed_at=date.today(),
        raw_components=components,
    )
