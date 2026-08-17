"""
MarketMaster Orchestrator — Coordinates specialist agents and debate.

Phase 3: Full agent chain with bull/bear debate framework.

The orchestrator:
1. Dispatches analysis to all specialist agents (macro, fundamental, technical, options, sentiment)
2. Collects structured evidence from each agent
3. Runs the bull/bear debate to synthesize a final recommendation
4. Runs the deterministic risk gate (which has FINAL AUTHORITY)

The orchestrator CANNOT bypass the risk gate. The risk gate has final
authority on whether a trade is executed.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from marketmaster.data.plane import DataPlane
from marketmaster.domain.models import DecisionEvidence, Opportunity
from marketmaster.engines.scoring import opportunity_score
from marketmaster.risk.gate import risk_gate
from marketmaster.config import settings
from marketmaster.agents.base import SpecialistAgent
from marketmaster.agents.debate import BullBearDebate, DebateResult


@dataclass
class AnalysisResult:
    """Result of orchestrating an analysis across agents."""
    symbol: str
    as_of: Optional[date] = None
    opportunity: Optional[Opportunity] = None
    debate: Optional[DebateResult] = None
    evidence: list[DecisionEvidence] = field(default_factory=list)
    risk_decision: Optional[Any] = None
    data_available: bool = True
    notes: list[str] = field(default_factory=list)
    agent_scores: dict[str, float] = field(default_factory=dict)


class MarketMasterOrchestrator:
    """
    Coordinates specialist agents for market analysis.

    Phase 3: Full agent chain with bull/bear debate.

    Flow:
    1. Resolve security via DataPlane
    2. Dispatch to all registered agents (parallel conceptually)
    3. Collect DecisionEvidence from each agent
    4. Run BullBearDebate to synthesize bull vs bear case
    5. Compute opportunity score from agent scores
    6. Run risk gate (deterministic, authoritative)
    """

    def __init__(self, db_session=None, agents: Optional[list[SpecialistAgent]] = None):
        self.plane = DataPlane(db_session) if db_session else None
        self.debate = BullBearDebate()

        # Default agent set for Phase 3
        if agents is not None:
            self.agents = agents
        else:
            # Lazy import to avoid circular dependencies
            self.agents = []
            try:
                from marketmaster.agents.macro import MacroAgent
                from marketmaster.agents.fundamental import FundamentalAgent
                from marketmaster.agents.technical import TechnicalAgent
                self.agents = [
                    MacroAgent(),
                    FundamentalAgent(),
                    TechnicalAgent(),
                ]
            except ImportError:
                pass

    def analyze(self, symbol: str, as_of: Optional[date] = None) -> AnalysisResult:
        """
        Analyze a symbol across all specialist agents.

        Returns a structured AnalysisResult with evidence from each agent,
        a bull/bear debate result, and a risk gate decision.
        """
        if as_of is None:
            as_of = date.today()

        result = AnalysisResult(symbol=symbol, as_of=as_of)

        # Check if we have data for this symbol
        if not self.plane:
            result.data_available = False
            result.notes.append("No database connection — running in stub mode")
            return result

        sec = self.plane.get_security_by_symbol(symbol)
        if not sec:
            result.data_available = False
            result.notes.append(f"Security not found: {symbol}")
            return result

        # Check for latest price data
        latest = self.plane.get_latest_price(sec.id, as_of)
        if not latest:
            result.data_available = False
            result.notes.append(f"No OHLCV data available for {symbol}")
            return result

        result.notes.append(f"Latest price: ${float(latest.close):.2f} on {latest.date}")

        # ── Dispatch to specialist agents ───────────────────────────────────
        for agent in self.agents:
            try:
                evidence = agent.analyze(
                    symbol=symbol,
                    security_id=sec.id,
                    as_of=as_of,
                    plane=self.plane,
                )
                result.evidence.append(evidence)

                # Collect agent scores
                for k, v in evidence.scores.items():
                    result.agent_scores[f"{evidence.agent}_{k}"] = v

            except Exception as e:
                result.notes.append(f"Agent {agent.name} error: {e}")

        # ── Run bull/bear debate ─────────────────────────────────────────────
        if result.evidence:
            result.debate = self.debate.run(
                symbol=symbol,
                evidence=result.evidence,
            )

            # Aggregate scores for opportunity score
            scores: dict[str, float] = {}
            for ev in result.evidence:
                for k, v in ev.scores.items():
                    scores[k] = v

            if scores:
                score = opportunity_score(scores)

                # Get current regime
                regime = "NEUTRAL"
                latest_regime = self.plane.get_latest_regime()
                if latest_regime:
                    regime = latest_regime.regime

                # Use debate confidence to adjust risk score
                risk_score = (1.0 - result.debate.confidence) * 100  # Low confidence = high risk

                result.opportunity = Opportunity(
                    symbol=symbol,
                    score=score,
                    strategy="pending",  # Strategy selection is Phase 4
                    regime=regime,
                    risk_score=risk_score,
                    expected_value=None,
                )

        return result

    def check_risk(
        self,
        position_risk_pct: float,
        daily_loss_pct: float,
    ):
        """
        Run the deterministic risk gate.

        The risk gate has FINAL AUTHORITY. No amount of bullish enthusiasm
        bypasses position limits, daily loss limits, or the live trading flag.
        """
        return risk_gate(
            position_risk_pct=position_risk_pct,
            max_position_risk_pct=settings.max_position_risk_pct,
            daily_loss_pct=daily_loss_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            live_trading_enabled=settings.enable_live_trading,
        )

    def get_full_analysis(self, symbol: str, as_of: Optional[date] = None) -> dict:
        """
        Get a complete analysis including debate, evidence, and risk assessment.

        Returns a dict suitable for API response.
        """
        result = self.analyze(symbol, as_of)

        response = {
            "symbol": result.symbol,
            "as_of": result.as_of.isoformat() if result.as_of else None,
            "data_available": result.data_available,
            "notes": result.notes,
        }

        if result.opportunity:
            response["opportunity"] = {
                "score": result.opportunity.score,
                "strategy": result.opportunity.strategy,
                "regime": result.opportunity.regime,
                "risk_score": result.opportunity.risk_score,
            }

        if result.debate:
            response["debate"] = {
                "bull_score": result.debate.bull_score,
                "bear_score": result.debate.bear_score,
                "net_score": result.debate.net_score,
                "winner": result.debate.winner,
                "confidence": result.debate.confidence,
                "summary": result.debate.summary,
                "contradictions": result.debate.contradictions,
                "key_risks": result.debate.key_risks,
            }

        response["agent_evidence"] = [
            {
                "agent": ev.agent,
                "observations": ev.observations,
                "scores": ev.scores,
                "bull_case": ev.bull_case,
                "bear_case": ev.bear_case,
                "risks": ev.risks,
                "data_quality": ev.data_quality,
                "confidence": ev.confidence,
                "recommended_actions": ev.recommended_actions,
            }
            for ev in result.evidence
        ]

        response["agent_scores"] = result.agent_scores

        return response
