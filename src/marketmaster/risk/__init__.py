"""
MarketMaster risk package.

Phase 5: Full deterministic risk engine with final authority over all trades.
"""

from marketmaster.risk.gate import risk_gate, RiskDecision as SimpleRiskDecision
from marketmaster.risk.engine import (
    RiskEngine, RiskDecision, RiskCheck, RiskLevel,
    PortfolioRiskState, KillSwitchState,
)

__all__ = [
    "risk_gate", "SimpleRiskDecision",
    "RiskEngine", "RiskDecision", "RiskCheck", "RiskLevel",
    "PortfolioRiskState", "KillSwitchState",
]
