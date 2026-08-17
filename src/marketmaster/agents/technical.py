"""
Technical Agent — Analyzes price action, trends, momentum, and volume.

Domain: OHLCV data, technical indicators, support/resistance, trend structure

The Technical Agent answers: What is the price telling us? It looks at
trend strength, momentum, volatility, and volume to assess whether
the current price action supports or contradicts the fundamental case.
"""

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

from marketmaster.agents.base import SpecialistAgent
from marketmaster.domain.models import DecisionEvidence
from marketmaster.engines.technical import compute_all_technical


class TechnicalAgent(SpecialistAgent):
    """Analyzes technical indicators and price action for a security."""

    def __init__(self, benchmark_symbol: str = "SPY"):
        super().__init__(
            name="technical",
            domain="technical",
            description="Analyzes price action, trends, momentum, volume, volatility",
        )
        self.benchmark_symbol = benchmark_symbol

    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        evidence = self._make_evidence()

        # ── Fetch OHLCV Data ──────────────────────────────────────────────────
        start = date(as_of.year - 2, 1, 1)  # ~2 years for SMA200
        bars = plane.get_ohlcv_daily(security_id, start_date=start, end_date=as_of)

        if len(bars) < 30:
            evidence.observations.append(f"Insufficient price data ({len(bars)} bars, need 30+)")
            evidence.data_quality = 0.0
            evidence.confidence = 0.0
            return evidence

        # Convert to pandas Series
        dates = [b.date for b in bars]
        highs = pd.Series([float(b.high) if b.high else np.nan for b in bars], index=dates)
        lows = pd.Series([float(b.low) if b.low else np.nan for b in bars], index=dates)
        closes = pd.Series([float(b.close) if b.close else np.nan for b in bars], index=dates)
        volumes = pd.Series([float(b.volume) if b.volume else 0 for b in bars], index=dates)

        close = float(closes.iloc[-1])
        evidence.observations.append(f"Latest close: ${close:.2f} on {dates[-1]}")

        # ── Benchmark ─────────────────────────────────────────────────────────
        benchmark_closes = None
        bench_sec = plane.get_security_by_symbol(self.benchmark_symbol)
        if bench_sec and bench_sec.id != security_id:
            bench_bars = plane.get_ohlcv_daily(bench_sec.id, start_date=start, end_date=as_of)
            if bench_bars:
                bench_dates = [b.date for b in bench_bars]
                benchmark_closes = pd.Series(
                    [float(b.close) if b.close else np.nan for b in bench_bars],
                    index=bench_dates,
                )

        # ── Compute All Technical Indicators ─────────────────────────────────
        indicators = compute_all_technical(highs, lows, closes, volumes, benchmark_closes)

        # ── Trend Analysis ───────────────────────────────────────────────────
        sma20 = indicators.get("sma_20")
        sma50 = indicators.get("sma_50")
        sma200 = indicators.get("sma_200")

        trend_score = 50.0
        if sma20 and sma20.value and sma50 and sma50.value:
            if sma20.value > sma50.value:
                evidence.observations.append("SMA20 above SMA50 — short-term uptrend")
                trend_score += 10
                evidence.bull_case.append("Price above short-term moving averages — uptrend intact")
            else:
                evidence.observations.append("SMA20 below SMA50 — short-term downtrend")
                trend_score -= 10
                evidence.bear_case.append("Price below short-term moving averages — downtrend")

        if sma200 and sma200.value:
            if close > sma200.value:
                evidence.observations.append("Price above 200-day SMA — long-term uptrend")
                trend_score += 15
                evidence.bull_case.append("Trading above 200-day SMA — long-term bullish structure")
            else:
                evidence.observations.append("Price below 200-day SMA — long-term downtrend")
                trend_score -= 15
                evidence.bear_case.append("Trading below 200-day SMA — long-term bearish structure")
                evidence.risks.append("Below 200-day SMA — structural weakness")

        if close and sma20 and sma20.value:
            sma50_str = f'${sma50.value:.2f}' if sma50 and sma50.value else 'N/A'
            evidence.observations.append(f'SMA20: ${sma20.value:.2f}, SMA50: {sma50_str}')

        evidence.scores["trend"] = self._safe_score(trend_score)

        # ── Momentum ──────────────────────────────────────────────────────────
        rsi = indicators.get("rsi_14")
        if rsi and rsi.value is not None:
            evidence.observations.append(f"RSI(14): {rsi.value:.1f} ({rsi.signal})")
            evidence.scores["rsi"] = rsi.value
            if rsi.value < 30:
                evidence.bull_case.append(f"RSI oversold at {rsi.value:.0f} — potential bounce")
            elif rsi.value > 70:
                evidence.bear_case.append(f"RSI overbought at {rsi.value:.0f} — potential pullback")
                evidence.risks.append("Overbought conditions — near-term pullback risk")

        macd_hist = indicators.get("macd_histogram")
        if macd_hist and macd_hist.value is not None:
            evidence.observations.append(f"MACD histogram: {macd_hist.value:.4f} ({macd_hist.signal})")
            if macd_hist.value > 0:
                evidence.scores["macd_momentum"] = 65.0
                evidence.bull_case.append("MACD histogram positive — bullish momentum")
            else:
                evidence.scores["macd_momentum"] = 35.0
                evidence.bear_case.append("MACD histogram negative — bearish momentum")

        mom = indicators.get("momentum_10")
        if mom and mom.value is not None:
            evidence.observations.append(f"10-day momentum: {mom.value:.2%}")
            evidence.scores["momentum"] = self._safe_score(50 + mom.value * 500)

        # ── Trend Strength ────────────────────────────────────────────────────
        adx = indicators.get("adx_14")
        if adx and adx.value is not None:
            evidence.observations.append(f"ADX(14): {adx.value:.1f} ({adx.signal})")
            evidence.scores["trend_strength"] = adx.value
            if adx.value > 25:
                evidence.bull_case.append(f"Strong trend (ADX={adx.value:.0f}) — momentum is real")
            elif adx.value < 20:
                evidence.observations.append("Weak trend (ADX < 20) — choppy/range-bound market")

        # ── Volatility ───────────────────────────────────────────────────────
        atr = indicators.get("atr_14")
        if atr and atr.value is not None and close > 0:
            atr_pct = (atr.value / close) * 100
            evidence.observations.append(f"ATR(14): ${atr.value:.2f} ({atr_pct:.1f}% of price)")
            if atr_pct > 4:
                evidence.risks.append(f"High volatility (ATR={atr_pct:.1f}%) — use smaller position size")
                evidence.scores["volatility"] = 30.0
            elif atr_pct < 1.5:
                evidence.scores["volatility"] = 75.0
            else:
                evidence.scores["volatility"] = 50.0

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_pos = indicators.get("bollinger_position")
        if bb_pos and bb_pos.value is not None:
            evidence.observations.append(f"Bollinger position: {bb_pos.value:.2f} (0=lower, 1=upper)")
            if bb_pos.value < 0.2:
                evidence.bull_case.append("Near lower Bollinger Band — potential mean reversion")
            elif bb_pos.value > 0.8:
                evidence.risks.append("Near upper Bollinger Band — extended, pullback risk")

        # ── Volume ───────────────────────────────────────────────────────────
        vol_ratio = indicators.get("volume_ratio")
        if vol_ratio and vol_ratio.value is not None:
            evidence.observations.append(f"Volume ratio: {vol_ratio.value:.2f}x average")
            if vol_ratio.value > 2.0:
                evidence.observations.append("Unusually high volume — significant interest/activity")
            if vol_ratio.value > 1.5 and close > (sma20.value if sma20 and sma20.value else close):
                evidence.bull_case.append("High volume confirming price strength")

        # ── Relative Strength ─────────────────────────────────────────────────
        rs = indicators.get("relative_strength_60")
        if rs and rs.value is not None:
            evidence.observations.append(f"Relative strength (60d vs {self.benchmark_symbol}): {rs.value:.2f} ({rs.signal})")
            evidence.scores["relative_strength"] = self._safe_score(rs.value * 50)
            if rs.value > 1.1:
                evidence.bull_case.append(f"Outperforming benchmark by {((rs.value - 1) * 100):.0f}%")
            elif rs.value < 0.9:
                evidence.bear_case.append(f"Underperforming benchmark by {((1 - rs.value) * 100):.0f}%")

        # ── Data Quality ─────────────────────────────────────────────────────
        if len(bars) >= 200:
            evidence.data_quality = 1.0
        elif len(bars) >= 50:
            evidence.data_quality = 0.7
        else:
            evidence.data_quality = 0.4

        # ── Confidence ───────────────────────────────────────────────────────
        non_neutral_scores = len([v for v in evidence.scores.values() if abs(v - 50) > 5])
        evidence.confidence = min(0.85, non_neutral_scores / 8.0)

        # ── Recommended Actions ───────────────────────────────────────────────
        avg_score = np.mean(list(evidence.scores.values())) if evidence.scores else 50.0
        if avg_score > 65:
            evidence.recommended_actions.append({"action": "technical_long", "strength": "high"})
        elif avg_score < 35:
            evidence.recommended_actions.append({"action": "technical_short", "strength": "high"})

        return evidence
