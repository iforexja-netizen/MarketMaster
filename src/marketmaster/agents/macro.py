"""
Macro Agent — Analyzes the macroeconomic environment and its impact on securities.

Domain: MCEI, liquidity, rates, yield curve, financial conditions, regime
Hierarchy: Money → Credit → Liquidity → Rates → Yield Curve → Financial
           Conditions → MCEI → Market Regime → Strategy Selection

The Macro Agent is the top of the analysis chain. It establishes the
macroeconomic regime that all other agents operate within.
"""

from datetime import date
from typing import Any, Optional

import numpy as np

from marketmaster.agents.base import SpecialistAgent
from marketmaster.domain.models import DecisionEvidence


class MacroAgent(SpecialistAgent):
    """Analyzes macro conditions and their implications for a security."""

    def __init__(self):
        super().__init__(
            name="macro",
            domain="macro",
            description="Analyzes MCEI, liquidity, rates, yield curve, and regime",
        )

    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        evidence = self._make_evidence()

        # ── MCEI Score & Regime ──────────────────────────────────────────────
        mcei = plane.get_latest_mcei()
        if mcei:
            score = float(mcei.score)
            regime = mcei.regime
            components = mcei.components if hasattr(mcei, 'components') else {}

            evidence.observations.append(f"MCEI score: {score:.1f} ({regime})")

            # Decompose: which components are bullish vs bearish
            bullish_comps = []
            bearish_comps = []
            for comp_name, comp_val in components.items():
                if isinstance(comp_val, (int, float)):
                    if comp_val >= 60:
                        bullish_comps.append(f"{comp_name} ({comp_val:.0f})")
                    elif comp_val <= 40:
                        bearish_comps.append(f"{comp_name} ({comp_val:.0f})")

            if bullish_comps:
                evidence.observations.append(f"Bullish components: {', '.join(bullish_comps[:5])}")
            if bearish_comps:
                evidence.observations.append(f"Bearish components: {', '.join(bearish_comps[:5])}")

            # Score: macro alignment with security (all securities get same macro score)
            evidence.scores["macro_alignment"] = score
            evidence.scores["mcei_regime_score"] = self._regime_to_score(regime)

            # Bull/bear case based on regime
            if score >= 60:
                evidence.bull_case.append(f"Macro environment is expansionary (MCEI={score:.0f}, {regime})")
                evidence.bull_case.append("Liquidity conditions supportive of risk assets")
                evidence.confidence = 0.7
            elif score >= 40:
                evidence.bull_case.append("Macro environment is neutral — no strong tailwind or headwind")
                evidence.confidence = 0.4
            else:
                evidence.bear_case.append(f"Macro environment is contractionary (MCEI={score:.0f}, {regime})")
                evidence.bear_case.append("Liquidity conditions adverse for risk assets")
                evidence.confidence = 0.7
        else:
            evidence.observations.append("No MCEI data available — cannot assess macro environment")
            evidence.data_quality = 0.0
            evidence.confidence = 0.0

        # ── Regime from regime_history ────────────────────────────────────────
        regime = plane.get_latest_regime()
        if regime:
            evidence.observations.append(f"Market regime: {regime.regime}")
            if regime.confidence:
                evidence.scores["regime_confidence"] = float(regime.confidence) * 100

            # Regime-specific risks
            if regime.regime in ("BEAR", "CRISIS"):
                evidence.risks.append(f"Market in {regime.regime} regime — elevated systematic risk")
            elif regime.regime == "TRANSITION_BEAR":
                evidence.risks.append("Regime transitioning to bear — reduce exposure")
            elif regime.regime == "TRANSITION_BULL":
                evidence.observations.append("Regime transitioning to bull — opportunity to add exposure")
        else:
            evidence.risks.append("No regime classification — macro state unknown")

        # ── Data Quality ─────────────────────────────────────────────────────
        if mcei:
            comp_count = len(components) if isinstance(components, dict) else 0
            evidence.data_quality = min(1.0, comp_count / 16.0)  # 16 total components
        else:
            evidence.data_quality = 0.0

        return evidence

    def _regime_to_score(self, regime: str) -> float:
        """Map regime name to a 0-100 score."""
        scores = {
            "STRONG_BULL": 90, "BULL": 70, "TRANSITION_BULL": 60,
            "NEUTRAL": 50,
            "TRANSITION_BEAR": 40, "BEAR": 25, "CRISIS": 10,
            "RECOVERY": 55,
            "STRONG_EXPANSION": 90, "EXPANSION": 70, "CONTRACTION": 25,
            "STRONG_CONTRACTION": 10,
        }
        return scores.get(regime, 50.0)
