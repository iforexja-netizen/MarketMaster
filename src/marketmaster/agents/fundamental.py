"""
Fundamental Agent — Analyzes financial statements and valuation.

Domain: Revenue, earnings, margins, growth, balance sheet, cash flow, valuation ratios

The Fundamental Agent answers: Is this company financially healthy and
reasonably valued? It uses point-in-time fundamentals (filing_date, not
report_date) to avoid look-ahead bias.
"""

from datetime import date
from typing import Any, Optional

import numpy as np

from marketmaster.agents.base import SpecialistAgent
from marketmaster.domain.models import DecisionEvidence


class FundamentalAgent(SpecialistAgent):
    """Analyzes fundamentals and valuation for a security."""

    def __init__(self):
        super().__init__(
            name="fundamental",
            domain="fundamental",
            description="Analyzes financials, valuation, growth, and quality",
        )

    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        evidence = self._make_evidence()

        # ── Get Latest Price ─────────────────────────────────────────────────
        latest_price = plane.get_latest_price(security_id, as_of)
        close = float(latest_price.close) if latest_price and latest_price.close else None

        if close:
            evidence.observations.append(f"Latest close: ${close:.2f} (as of {latest_price.date})")

        # ── Get Fundamentals ──────────────────────────────────────────────────
        fundamentals = plane.get_fundamentals(security_id=security_id, realtime_date=as_of)

        if not fundamentals:
            evidence.observations.append("No fundamental data available")
            evidence.data_quality = 0.0
            evidence.confidence = 0.0
            return evidence

        latest_fund = fundamentals[0]
        items = latest_fund.items if hasattr(latest_fund, 'items') else {}

        if not items:
            evidence.observations.append("Fundamental data exists but items dict is empty")
            evidence.data_quality = 0.1
            evidence.confidence = 0.1
            return evidence

        evidence.observations.append(
            f"Latest filing: {latest_fund.report_date} (period: {latest_fund.period_type}), "
            f"filed: {latest_fund.filing_date or 'unknown'}"
        )

        # ── Compute Fundamental Factors ───────────────────────────────────────
        prior_items = fundamentals[1].items if len(fundamentals) > 1 and hasattr(fundamentals[1], 'items') else {}
        shares = items.get("CommonStockSharesOutstanding")
        market_cap = (close * shares) if (close and shares) else None

        # Valuation
        revenue = items.get("Revenues")
        net_income = items.get("NetIncomeLoss")
        equity = items.get("StockholdersEquity")
        assets = items.get("Assets")
        liabilities = items.get("Liabilities")
        eps = items.get("EarningsPerShareBasic")
        operating_income = items.get("OperatingIncomeLoss")
        lt_debt = items.get("LongTermDebt")
        cash = items.get("CashAndCashEquivalentsAtCarryingValue")

        # ── Profitability Assessment ─────────────────────────────────────────
        if net_income is not None and equity and equity > 0:
            roe = net_income / equity
            evidence.scores["roe"] = self._safe_score(self._pct_to_score(roe, 0.15))  # 15% ROE = good
            evidence.observations.append(f"ROE: {roe:.1%}")
            if roe > 0.15:
                evidence.bull_case.append(f"Strong ROE of {roe:.1%} — efficient capital allocation")
            elif roe < 0.05 and roe > 0:
                evidence.bear_case.append(f"Low ROE of {roe:.1%} — poor capital efficiency")

        if net_income is not None and revenue and revenue > 0:
            net_margin = net_income / revenue
            evidence.scores["net_margin"] = self._safe_score(self._pct_to_score(net_margin, 0.10))
            evidence.observations.append(f"Net margin: {net_margin:.1%}")

        # ── Valuation Assessment ─────────────────────────────────────────────
        if close and eps and eps > 0:
            pe = close / eps
            evidence.scores["valuation"] = self._safe_score(self._pe_to_score(pe))
            evidence.observations.append(f"P/E: {pe:.1f}")
            if pe < 15:
                evidence.bull_case.append(f"Attractive P/E of {pe:.1f} — undervalued relative to earnings")
            elif pe > 35:
                evidence.bear_case.append(f"High P/E of {pe:.1f} — priced for significant growth")

        if market_cap and revenue and revenue > 0:
            ps = market_cap / revenue
            evidence.observations.append(f"P/S: {ps:.1f}")

        # ── Growth Assessment ────────────────────────────────────────────────
        if prior_items:
            prior_revenue = prior_items.get("Revenues")
            prior_ni = prior_items.get("NetIncomeLoss")

            if revenue and prior_revenue and prior_revenue > 0:
                rev_growth = (revenue / prior_revenue) - 1
                evidence.scores["revenue_growth"] = self._safe_score(self._growth_to_score(rev_growth))
                evidence.observations.append(f"Revenue growth YoY: {rev_growth:.1%}")
                if rev_growth > 0.15:
                    evidence.bull_case.append(f"Strong revenue growth of {rev_growth:.1%}")
                elif rev_growth < 0:
                    evidence.bear_case.append(f"Revenue declining ({rev_growth:.1%})")

            if net_income is not None and prior_ni is not None and prior_ni > 0:
                ni_growth = (net_income / prior_ni) - 1
                evidence.scores["earnings_growth"] = self._safe_score(self._growth_to_score(ni_growth))
                evidence.observations.append(f"Earnings growth YoY: {ni_growth:.1%}")

        # ── Leverage Assessment ──────────────────────────────────────────────
        if lt_debt is not None and equity and equity > 0:
            de = lt_debt / equity
            evidence.scores["leverage"] = self._safe_score(self._de_to_score(de))
            evidence.observations.append(f"Debt/Equity: {de:.2f}")
            if de > 2.0:
                evidence.bear_case.append(f"High leverage (D/E={de:.1f}) — balance sheet risk")
                evidence.risks.append("Elevated debt levels — vulnerable to rate increases")
            elif de < 0.3:
                evidence.bull_case.append(f"Low leverage (D/E={de:.1f}) — strong balance sheet")

        # ── Balance Sheet Health ─────────────────────────────────────────────
        if assets and liabilities and assets > 0:
            equity_ratio = 1 - (liabilities / assets)
            evidence.observations.append(f"Equity ratio: {equity_ratio:.1%}")
            if equity_ratio < 0.3:
                evidence.risks.append("Low equity ratio — balance sheet vulnerability")

        # ── Data Quality ─────────────────────────────────────────────────────
        expected_items = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities", "StockholdersEquity"]
        filled = sum(1 for k in expected_items if items.get(k) is not None)
        evidence.data_quality = filled / len(expected_items)

        # ── Confidence ───────────────────────────────────────────────────────
        score_count = len([v for v in evidence.scores.values() if v != 50.0])
        evidence.confidence = min(0.9, score_count / 6.0)

        # ── Recommended Actions ───────────────────────────────────────────────
        avg_score = np.mean(list(evidence.scores.values())) if evidence.scores else 50.0
        if avg_score > 65:
            evidence.recommended_actions.append({"action": "fundamental_long", "strength": "high"})
        elif avg_score < 35:
            evidence.recommended_actions.append({"action": "fundamental_short", "strength": "high"})

        return evidence

    def _pct_to_score(self, pct: float, benchmark: float = 0.10) -> float:
        """Convert a percentage (e.g., margin) to a 0-100 score."""
        if pct <= 0:
            return max(0, 50 + pct * 200)  # Negative margins score below 50
        return min(100, 50 + (pct / benchmark - 1) * 50)

    def _pe_to_score(self, pe: float) -> float:
        """Convert P/E ratio to score (lower P/E = higher score, but not linearly)."""
        if pe <= 0:
            return 20  # Negative earnings
        if pe <= 10:
            return 90
        if pe <= 15:
            return 75
        if pe <= 20:
            return 60
        if pe <= 30:
            return 40
        if pe <= 50:
            return 25
        return 15

    def _growth_to_score(self, growth: float) -> float:
        """Convert growth rate to score."""
        if growth <= 0:
            return max(10, 40 + growth * 100)
        if growth <= 0.05:
            return 50
        if growth <= 0.10:
            return 65
        if growth <= 0.20:
            return 80
        if growth <= 0.30:
            return 90
        return 95

    def _de_to_score(self, de: float) -> float:
        """Convert D/E to score (lower = better)."""
        if de <= 0:
            return 95
        if de <= 0.3:
            return 85
        if de <= 0.5:
            return 75
        if de <= 1.0:
            return 55
        if de <= 2.0:
            return 35
        return 20
