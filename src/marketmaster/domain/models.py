from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class DecisionEvidence:
    agent: str
    timestamp: datetime
    observations: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    bull_case: list[str] = field(default_factory=list)
    bear_case: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    data_quality: float = 0.0
    confidence: float = 0.0
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class Opportunity:
    symbol: str
    score: float
    strategy: str
    regime: str
    risk_score: float
    expected_value: float | None = None
