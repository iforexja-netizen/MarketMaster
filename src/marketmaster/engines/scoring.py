DEFAULT_WEIGHTS = {
    "fundamental_quality": .15, "valuation": .10, "technical_structure": .15,
    "momentum": .10, "macro_alignment": .10, "options": .10,
    "catalyst": .05, "sentiment": .05, "liquidity": .05, "risk_reward": .15,
}

def opportunity_score(scores: dict[str, float], weights=None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    if total <= 0: raise ValueError("Weights must sum to a positive value")
    return sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights) / total
