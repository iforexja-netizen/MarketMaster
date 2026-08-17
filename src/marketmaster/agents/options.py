"""
Options Agent — Analyzes options chains, implied volatility, and Greeks.

Domain: Option chains, IV rank/percentile, put/call ratio, skew, gamma exposure

The Options Agent answers: What is the options market pricing in?
Options are forward-looking — they reveal market expectations about
volatility, direction, and tail risk.
"""

from datetime import date
from typing import Any, Optional

import numpy as np

from marketmaster.agents.base import SpecialistAgent
from marketmaster.domain.models import DecisionEvidence


class OptionsAgent(SpecialistAgent):
    """Analyzes options data for a security."""

    def __init__(self):
        super().__init__(
            name="options",
            domain="options",
            description="Analyzes IV, put/call ratio, skew, and options sentiment",
        )

    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        evidence = self._make_evidence()

        # ── Fetch Option Chain ────────────────────────────────────────────────
        try:
            options = plane.get_option_chain(security_id, as_of_date=as_of)
        except AttributeError:
            options = None
        except Exception:
            options = None

        if not options:
            evidence.observations.append("No options data available")
            evidence.data_quality = 0.0
            evidence.confidence = 0.0
            return evidence

        calls = [o for o in options if hasattr(o, 'option_type') and o.option_type == 'call']
        puts = [o for o in options if hasattr(o, 'option_type') and o.option_type == 'put']

        if not calls and not puts:
            evidence.observations.append("Options data exists but no calls/puts found")
            evidence.data_quality = 0.1
            evidence.confidence = 0.0
            return evidence

        evidence.observations.append(
            f"Option chain: {len(calls)} calls, {len(puts)} puts "
            f"(as of {as_of})"
        )

        # ── Put/Call Ratio ────────────────────────────────────────────────────
        call_oi = sum(float(o.open_interest) for o in calls if hasattr(o, 'open_interest') and o.open_interest)
        put_oi = sum(float(o.open_interest) for o in puts if hasattr(o, 'open_interest') and o.open_interest)
        call_vol = sum(float(o.volume) for o in calls if hasattr(o, 'volume') and o.volume)
        put_vol = sum(float(o.volume) for o in puts if hasattr(o, 'volume') and o.volume)

        if put_oi > 0 and call_oi > 0:
            pcr_oi = put_oi / call_oi
            evidence.observations.append(f"Put/Call OI ratio: {pcr_oi:.2f}")
            evidence.scores["put_call_ratio"] = self._pcr_to_score(pcr_oi)

            if pcr_oi > 1.5:
                evidence.bear_case.append(f"High put/call ratio ({pcr_oi:.1f}) — hedging or bearish positioning")
                evidence.risks.append("Elevated put buying — institutional hedging or bearish sentiment")
            elif pcr_oi < 0.5:
                evidence.bull_case.append(f"Low put/call ratio ({pcr_oi:.1f}) — call-heavy positioning, bullish sentiment")

        if put_vol > 0 and call_vol > 0:
            pcr_vol = put_vol / call_vol
            evidence.observations.append(f"Put/Call volume ratio: {pcr_vol:.2f}")

        # ── Implied Volatility ───────────────────────────────────────────────
        ivs = []
        for o in options:
            if hasattr(o, 'implied_volatility') and o.implied_volatility:
                ivs.append(float(o.implied_volatility))

        if ivs:
            avg_iv = float(np.mean(ivs))
            evidence.observations.append(f"Average IV: {avg_iv:.1%}")

            # IV percentile (would need historical IV data for proper percentile)
            # For now, use a simple scale
            if avg_iv > 0.50:
                evidence.risks.append(f"Elevated IV ({avg_iv:.0%}) — options pricing in high uncertainty")
                evidence.scores["implied_volatility"] = 25.0
            elif avg_iv > 0.30:
                evidence.observations.append("Moderate IV — normal uncertainty range")
                evidence.scores["implied_volatility"] = 50.0
            else:
                evidence.bull_case.append(f"Low IV ({avg_iv:.0%}) — market calm, cheap option premiums")
                evidence.scores["implied_volatility"] = 75.0

        # ── IV Skew (if we have strikes) ──────────────────────────────────────
        atm_ivs = []
        otm_put_ivs = []
        otm_call_ivs = []

        # Get the ATM price for strike comparison
        latest_price = plane.get_latest_price(security_id, as_of)
        spot = float(latest_price.close) if latest_price and latest_price.close else None

        if spot:
            for o in options:
                if not hasattr(o, 'strike') or not o.strike or not hasattr(o, 'implied_volatility') or not o.implied_volatility:
                    continue
                strike = float(o.strike)
                iv = float(o.implied_volatility)
                moneyness = strike / spot

                if 0.97 <= moneyness <= 1.03:
                    atm_ivs.append(iv)
                elif moneyness < 0.90:
                    otm_put_ivs.append(iv)
                elif moneyness > 1.10:
                    otm_call_ivs.append(iv)

            if atm_ivs and otm_put_ivs:
                atm_iv = float(np.mean(atm_ivs))
                otm_put_iv = float(np.mean(otm_put_ivs))
                skew = otm_put_iv - atm_iv
                evidence.observations.append(f"IV skew (OTM put - ATM): {skew:.3f}")
                evidence.scores["iv_skew"] = self._safe_score(50 + skew * 100)

                if skew > 0.05:
                    evidence.bear_case.append(f"Steep put skew — market pricing in downside tail risk")
                    evidence.risks.append("Elevated put skew — crash protection being bought")
                elif skew < -0.02:
                    evidence.bull_case.append("Inverted skew — call demand, bullish positioning")

        # ── Gamma Exposure (simplified) ──────────────────────────────────────
        if spot and calls:
            # Net gamma: calls positive, puts negative (simplified)
            total_gamma = 0
            for o in calls:
                if hasattr(o, 'gamma') and o.gamma and hasattr(o, 'open_interest') and o.open_interest:
                    total_gamma += float(o.gamma) * float(o.open_interest) * 100
            for o in puts:
                if hasattr(o, 'gamma') and o.gamma and hasattr(o, 'open_interest') and o.open_interest:
                    total_gamma -= float(o.gamma) * float(o.open_interest) * 100

            if total_gamma != 0:
                evidence.observations.append(f"Net gamma exposure: {total_gamma:.0f}")
                if total_gamma > 0:
                    evidence.observations.append("Positive net gamma — dealer hedging tends to dampen volatility")
                else:
                    evidence.risks.append("Negative net gamma — dealer hedging can amplify volatility moves")

        # ── Data Quality ─────────────────────────────────────────────────────
        if len(options) > 50:
            evidence.data_quality = 0.9
        elif len(options) > 10:
            evidence.data_quality = 0.6
        else:
            evidence.data_quality = 0.3

        evidence.confidence = min(0.7, evidence.data_quality * 0.8)

        return evidence

    def _pcr_to_score(self, pcr: float) -> float:
        """
        Convert put/call ratio to score.
        Low PCR = bullish (call-heavy), high PCR = bearish (put-heavy).
        """
        if pcr < 0.4:
            return 80.0
        if pcr < 0.7:
            return 65.0
        if pcr < 1.0:
            return 55.0
        if pcr < 1.3:
            return 45.0
        if pcr < 2.0:
            return 30.0
        return 15.0
