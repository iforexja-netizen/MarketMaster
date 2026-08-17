"""
MarketMaster agents package.

Phase 3: Full specialist agent chain with bull/bear debate framework.
"""

from marketmaster.agents.base import SpecialistAgent
from marketmaster.agents.orchestrator import MarketMasterOrchestrator, AnalysisResult
from marketmaster.agents.debate import BullBearDebate, DebateResult, DebateArgument

# Specialist agents
from marketmaster.agents.macro import MacroAgent
from marketmaster.agents.fundamental import FundamentalAgent
from marketmaster.agents.technical import TechnicalAgent
from marketmaster.agents.options import OptionsAgent
from marketmaster.agents.sentiment import SentimentAgent

__all__ = [
    "SpecialistAgent",
    "MarketMasterOrchestrator",
    "AnalysisResult",
    "BullBearDebate",
    "DebateResult",
    "DebateArgument",
    "MacroAgent",
    "FundamentalAgent",
    "TechnicalAgent",
    "OptionsAgent",
    "SentimentAgent",
]
