"""
Specialist Agent Base Class — Common interface for all MarketMaster agents.

Every specialist agent inherits from this base and implements `analyze()`.
Agents are pure intelligence: they receive data, reason about it, and return
structured evidence. They never execute trades or bypass the risk gate.

The evidence model is the contract between agents and the orchestrator:
every agent returns observations, scores, bull case, bear case, risks,
data quality assessment, and confidence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from marketmaster.domain.models import DecisionEvidence


class SpecialistAgent(ABC):
    """
    Base class for all specialist agents.

    Each agent:
    1. Has a name and domain (macro, fundamental, technical, etc.)
    2. Receives a symbol, as_of date, and a DataPlane reference
    3. Pulls relevant data from the DataPlane (point-in-time)
    4. Applies domain-specific analysis
    5. Returns structured DecisionEvidence

    Agents are read-only: they never write to the database.
    The orchestrator collects evidence and writes to the decision log.
    """

    def __init__(self, name: str, domain: str, description: str = ""):
        self.name = name
        self.domain = domain
        self.description = description

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        """
        Analyze a security and return structured evidence.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            security_id: Internal security ID for DB lookups
            as_of: Point-in-time date for analysis
            plane: DataPlane instance for data access

        Returns:
            DecisionEvidence with observations, scores, bull/bear case, risks
        """
        pass

    def _make_evidence(self, **kwargs) -> DecisionEvidence:
        """Helper to create evidence with agent name and timestamp."""
        defaults = {
            "agent": self.name,
            "timestamp": datetime.now(timezone.utc),
            "observations": [],
            "scores": {},
            "bull_case": [],
            "bear_case": [],
            "risks": [],
            "data_quality": 0.0,
            "confidence": 0.0,
            "recommended_actions": [],
        }
        defaults.update(kwargs)
        return DecisionEvidence(**defaults)

    def _safe_score(self, value: Optional[float], min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Clamp a value to a valid score range."""
        if value is None:
            return 50.0  # Neutral when no data
        return float(max(min_val, min(max_val, value)))
