"""
Bull/Bear Debate Framework

The debate framework is how MarketMaster synthesizes multiple agent analyses
into a final recommendation. Instead of a simple weighted average, it runs a
structured debate:

1. Bull camp: agents with bullish evidence argue for the long side
2. Bear camp: agents with bearish evidence argue for the short side
3. Each camp presents its strongest evidence
4. Cross-examination: each camp challenges the other's weak points
5. Verdict: weighted scoring with confidence adjustment based on agreement

This is more realistic than naive averaging because:
- When all agents agree, confidence is high (strong signal)
- When agents disagree, confidence is low (uncertain, reduce position)
- The strongest evidence from each side gets more weight
- Contradictions are explicitly flagged
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from marketmaster.domain.models import DecisionEvidence


@dataclass
class DebateArgument:
    """A single argument in the debate."""
    side: str  # "bull" or "bear"
    agent: str  # which agent made this argument
    argument: str
    evidence_strength: float  # 0-1, how strong is this argument
    challenged_by: list[str] = field(default_factory=list)  # counter-arguments
    survives_cross_examination: bool = True


@dataclass
class DebateResult:
    """Result of a bull/bear debate."""
    symbol: str
    bull_score: float  # 0-100
    bear_score: float  # 0-100
    net_score: float  # bull_score - bear_score (-100 to +100)
    winner: str  # "bull", "bear", or "split"
    confidence: float  # 0-1, based on agreement among agents
    bull_arguments: list[DebateArgument] = field(default_factory=list)
    bear_arguments: list[DebateArgument] = field(default_factory=list)
    cross_examination_notes: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    summary: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BullBearDebate:
    """
    Orchestrates a structured debate between bull and bear cases.

    Usage:
        debate = BullBearDebate()
        result = debate.run(symbol="AAPL", evidence=[e1, e2, e3, ...])
    """

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def run(
        self,
        symbol: str,
        evidence: list[DecisionEvidence],
        weights: Optional[dict[str, float]] = None,
    ) -> DebateResult:
        """
        Run a bull/bear debate over the evidence from multiple agents.

        Args:
            symbol: Security symbol
            evidence: List of DecisionEvidence from specialist agents
            weights: Optional per-agent weights (default: equal weight)

        Returns:
            DebateResult with scores, arguments, and verdict
        """
        if not evidence:
            return DebateResult(
                symbol=symbol,
                bull_score=50.0,
                bear_score=50.0,
                net_score=0.0,
                winner="split",
                confidence=0.0,
                summary="No evidence available for debate.",
            )

        # Default weights: equal across all agents
        if weights is None:
            weights = {ev.agent: 1.0 / len(evidence) for ev in evidence}

        # ── Phase 1: Gather Arguments ────────────────────────────────────────
        bull_arguments: list[DebateArgument] = []
        bear_arguments: list[DebateArgument] = []

        for ev in evidence:
            agent_weight = weights.get(ev.agent, 1.0 / len(evidence))

            # Bull case arguments
            for arg in ev.bull_case:
                # Strength based on agent confidence and data quality
                strength = ev.confidence * ev.data_quality * agent_weight
                bull_arguments.append(DebateArgument(
                    side="bull",
                    agent=ev.agent,
                    argument=arg,
                    evidence_strength=strength,
                ))

            # Bear case arguments
            for arg in ev.bear_case:
                strength = ev.confidence * ev.data_quality * agent_weight
                bear_arguments.append(DebateArgument(
                    side="bear",
                    agent=ev.agent,
                    argument=arg,
                    evidence_strength=strength,
                ))

        # ── Phase 2: Score Each Side ─────────────────────────────────────────
        # Use agent scores directly for quantitative assessment
        bull_score = self._compute_side_score(evidence, "bull", weights)
        bear_score = self._compute_side_score(evidence, "bear", weights)

        # Normalize: bear_score is inverted (high bear = bad)
        # bull_score already 0-100, bear_score already 0-100 (high = bearish)
        # Adjust so they sum to 100
        total = bull_score + bear_score
        if total > 0:
            bull_score = (bull_score / total) * 100
            bear_score = (bear_score / total) * 100

        # ── Phase 3: Cross-Examination ──────────────────────────────────────
        cross_notes, contradictions = self._cross_examine(bull_arguments, bear_arguments, evidence)

        # Filter arguments that survive cross-examination
        for arg in bull_arguments:
            if arg.challenged_by:
                # If challenged by strong counter-arguments, mark as weakened
                counter_strength = max(
                    (ba.evidence_strength for ba in bear_arguments if ba.argument in arg.challenged_by),
                    default=0,
                )
                if counter_strength > arg.evidence_strength:
                    arg.survives_cross_examination = False

        for arg in bear_arguments:
            if arg.challenged_by:
                counter_strength = max(
                    (ba.evidence_strength for ba in bull_arguments if ba.argument in arg.challenged_by),
                    default=0,
                )
                if counter_strength > arg.evidence_strength:
                    arg.survives_cross_examination = False

        # ── Phase 4: Confidence Assessment ───────────────────────────────────
        confidence = self._assess_confidence(evidence, bull_arguments, bear_arguments)

        # ── Phase 5: Verdict ─────────────────────────────────────────────────
        net = bull_score - bear_score
        if net > 15:
            winner = "bull"
        elif net < -15:
            winner = "bear"
        else:
            winner = "split"

        # ── Key Risks ────────────────────────────────────────────────────────
        key_risks = []
        for ev in evidence:
            for risk in ev.risks:
                key_risks.append(f"[{ev.agent}] {risk}")

        # Remove duplicates
        key_risks = list(dict.fromkeys(key_risks))

        # ── Summary ──────────────────────────────────────────────────────────
        surviving_bull = [a for a in bull_arguments if a.survives_cross_examination]
        surviving_bear = [a for a in bear_arguments if a.survives_cross_examination]

        summary = self._generate_summary(
            symbol, winner, bull_score, bear_score, confidence,
            surviving_bull, surviving_bear, contradictions,
        )

        return DebateResult(
            symbol=symbol,
            bull_score=bull_score,
            bear_score=bear_score,
            net_score=net,
            winner=winner,
            confidence=confidence,
            bull_arguments=bull_arguments,
            bear_arguments=bear_arguments,
            cross_examination_notes=cross_notes,
            contradictions=contradictions,
            key_risks=key_risks,
            summary=summary,
        )

    def _compute_side_score(
        self,
        evidence: list[DecisionEvidence],
        side: str,
        weights: dict[str, float],
    ) -> float:
        """
        Compute the aggregate score for one side of the debate.

        Uses each agent's quantitative scores, weighted by agent weight
        and confidence.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for ev in evidence:
            agent_weight = weights.get(ev.agent, 0.0)
            confidence_weight = max(0.1, ev.confidence)  # Even 0 confidence gets some weight
            combined_weight = agent_weight * confidence_weight * max(0.1, ev.data_quality)

            # Get the average score from this agent
            if ev.scores:
                agent_scores = list(ev.scores.values())
                avg_score = float(np.mean(agent_scores))
            else:
                avg_score = 50.0

            if side == "bull":
                # Bull score: higher agent scores = more bullish
                weighted_sum += avg_score * combined_weight
            else:
                # Bear score: lower agent scores = more bearish (invert)
                weighted_sum += (100 - avg_score) * combined_weight

            total_weight += combined_weight

        if total_weight == 0:
            return 50.0

        return weighted_sum / total_weight

    def _cross_examine(
        self,
        bull_args: list[DebateArgument],
        bear_args: list[DebateArgument],
        evidence: list[DecisionEvidence],
    ) -> tuple[list[str], list[str]]:
        """
        Cross-examination: identify contradictions and challenges.

        Returns (cross_examination_notes, contradictions).
        """
        cross_notes = []
        contradictions = []

        # Check for agents with contradictory scores
        agent_scores = {}
        for ev in evidence:
            if ev.scores:
                avg = float(np.mean(list(ev.scores.values())))
                agent_scores[ev.agent] = avg

        # Find agents that strongly disagree
        if len(agent_scores) >= 2:
            scores = list(agent_scores.values())
            score_range = max(scores) - min(scores)

            if score_range > 30:
                # Find the disagreeing agents
                max_agent = max(agent_scores, key=agent_scores.get)
                min_agent = min(agent_scores, key=agent_scores.get)
                contradictions.append(
                    f"Agent disagreement: {max_agent} is bullish ({agent_scores[max_agent]:.0f}) "
                    f"while {min_agent} is bearish ({agent_scores[min_agent]:.0f})"
                )

        # Cross-examine bull arguments against bear evidence
        for bull in bull_args:
            for bear in bear_args:
                if bear.agent != bull.agent:
                    # If both sides have strong arguments from different agents,
                    # they challenge each other
                    if bull.evidence_strength > 0.3 and bear.evidence_strength > 0.3:
                        # Check if they're on the same topic (simplified heuristic)
                        bull_words = set(bull.argument.lower().split())
                        bear_words = set(bear.argument.lower().split())
                        overlap = bull_words & bear_words
                        if len(overlap) >= 2:  # Some shared keywords
                            bull.challenged_by.append(bear.argument)
                            bear.challenged_by.append(bull.argument)
                            cross_notes.append(
                                f"Bull ({bull.agent}) vs Bear ({bear.agent}): conflicting evidence on "
                                f"shared topic"
                            )

        # Flag data quality issues
        for ev in evidence:
            if ev.data_quality < 0.3:
                cross_notes.append(
                    f"{ev.agent} has low data quality ({ev.data_quality:.0%}) — "
                    f"arguments from this agent are weakened"
                )

        return cross_notes, contradictions

    def _assess_confidence(
        self,
        evidence: list[DecisionEvidence],
        bull_args: list[DebateArgument],
        bear_args: list[DebateArgument],
    ) -> float:
        """
        Assess confidence based on agent agreement.

        High confidence: all agents agree (all bullish or all bearish)
        Low confidence: agents disagree (mixed signals)
        """
        if not evidence:
            return 0.0

        # Compute average agent score
        agent_avg_scores = []
        for ev in evidence:
            if ev.scores:
                avg = float(np.mean(list(ev.scores.values())))
                agent_avg_scores.append(avg)

        if not agent_avg_scores:
            return 0.0

        # Agreement = inverse of dispersion
        mean_score = float(np.mean(agent_avg_scores))
        std_score = float(np.std(agent_avg_scores))

        # High std = disagreement = low confidence
        agreement = max(0.0, 1.0 - (std_score / 50.0))  # 50 std = total disagreement

        # Also factor in individual agent confidence
        avg_agent_confidence = float(np.mean([ev.confidence for ev in evidence]))

        # Data quality factor
        avg_data_quality = float(np.mean([ev.data_quality for ev in evidence]))

        # Distance from neutral (50) — stronger signals = more confidence
        signal_strength = abs(mean_score - 50) / 50.0

        confidence = agreement * avg_agent_confidence * avg_data_quality * (0.5 + 0.5 * signal_strength)

        return float(min(1.0, confidence))

    def _generate_summary(
        self,
        symbol: str,
        winner: str,
        bull_score: float,
        bear_score: float,
        confidence: float,
        bull_args: list[DebateArgument],
        bear_args: list[DebateArgument],
        contradictions: list[str],
    ) -> str:
        """Generate a human-readable debate summary."""
        if winner == "bull":
            verdict = f"BULLISH — bull case wins ({bull_score:.0f} vs {bear_score:.0f})"
        elif winner == "bear":
            verdict = f"BEARISH — bear case wins ({bear_score:.0f} vs {bull_score:.0f})"
        else:
            verdict = f"NEUTRAL — split verdict ({bull_score:.0f} vs {bear_score:.0f})"

        lines = [
            f"{symbol} Debate Verdict: {verdict}",
            f"Confidence: {confidence:.0%}",
        ]

        if bull_args:
            lines.append(f"\nBull case ({len(bull_args)} arguments):")
            for arg in bull_args[:3]:  # Top 3
                marker = "✓" if arg.survives_cross_examination else "✗"
                lines.append(f"  {marker} [{arg.agent}] {arg.argument}")

        if bear_args:
            lines.append(f"\nBear case ({len(bear_args)} arguments):")
            for arg in bear_args[:3]:
                marker = "✓" if arg.survives_cross_examination else "✗"
                lines.append(f"  {marker} [{arg.agent}] {arg.argument}")

        if contradictions:
            lines.append(f"\nContradictions:")
            for c in contradictions:
                lines.append(f"  ⚠ {c}")

        return "\n".join(lines)
