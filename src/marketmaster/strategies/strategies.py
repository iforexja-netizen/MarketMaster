"""
MarketMaster Strategy Library — 16 regime-aware trading strategies.

Each strategy is transparent: it returns a score, reasoning, entry/target/stop
levels, and position size suggestion. Strategies are selected based on the
current market regime.

Strategy categories:
  Trend (4):  Trend Following, Momentum, Breakout, Earnings Momentum
  Mean Rev (3): Mean Reversion, Pairs Trading, RSI Reversal
  Value (3):  Value, Quality, Growth
  Defensive (3): Low Volatility, Defensive, Options Collar
  Macro (2): Sector Rotation, Macro Driven
  Sizing (1): Risk Parity

Regime mapping:
  STRONG_BULL: Trend Following, Momentum, Growth, Breakout, Sector Rotation
  BULL: Trend Following, Momentum, Value, Quality, Earnings Momentum
  TRANSITION_BULL: Value, Quality, Sector Rotation, Mean Reversion
  NEUTRAL: Mean Reversion, Pairs Trading, RSI Reversal, Value, Quality
  TRANSITION_BEAR: Low Volatility, Defensive, Value, Quality
  BEAR: Low Volatility, Defensive, Options Collar
  CRISIS: Defensive, Options Collar, Risk Parity
  RECOVERY: Trend Following, Value, Growth, Sector Rotation
"""

from datetime import date
from typing import Any, Optional
import numpy as np

from marketmaster.strategies.base import Strategy, SignalDirection, TradeSignal, StrategyConfig


# ============================================================================
# TREND STRATEGIES
# ============================================================================

class TrendFollowingStrategy(Strategy):
    """Buy securities in strong uptrends, ride the trend."""
    def __init__(self):
        super().__init__(
            name="trend_following",
            description="Follows established uptrends using moving averages and ADX",
            applicable_regimes=["STRONG_BULL", "BULL", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})
        debate_scores = self._get_debate_scores(debate)

        score = 50.0
        reasoning = []

        trend_score = tech.get("trend", 50)
        adx = tech.get("trend_strength", 20)
        rs = tech.get("relative_strength", 50)
        sma_relation = market_data.get("price", 0) > market_data.get("sma200", 0) if market_data.get("sma200") else True

        if sma_relation:
            score += 10
            reasoning.append("Price above 200-day SMA")
        else:
            score -= 10
            reasoning.append("Price below 200-day SMA")

        if adx > 25:
            score += 15
            reasoning.append(f"Strong trend (ADX={adx:.0f})")
        elif adx < 20:
            score -= 5
            reasoning.append("Weak trend — no clear direction")

        score += (trend_score - 50) * 0.5
        score += (rs - 50) * 0.3

        if debate_scores.get("net_score", 0) > 15:
            score += 5
            reasoning.append("Debate confirms bullish case")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.85, (adx / 50) * 0.5 + debate_scores.get("confidence", 0.3) * 0.5)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"trend": trend_score, "adx": adx, "rs": rs}, regime)


class MomentumStrategy(Strategy):
    """Buy securities with the strongest recent momentum."""
    def __init__(self):
        super().__init__(
            name="momentum",
            description="Buys securities with strong price momentum",
            applicable_regimes=["STRONG_BULL", "BULL"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []

        mom = tech.get("momentum", 50)
        rsi = tech.get("rsi", 50)
        macd = tech.get("macd_momentum", 50)

        score += (mom - 50) * 0.4
        score += (rsi - 50) * 0.3
        score += (macd - 50) * 0.3

        if rsi > 70:
            score -= 5
            reasoning.append("RSI overbought — momentum may be exhausting")
        elif rsi > 55:
            score += 5
            reasoning.append("RSI in momentum zone (55-70)")

        reasoning.append(f"10-day momentum score: {mom:.0f}")
        reasoning.append(f"MACD momentum: {macd:.0f}")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.8, abs(score - 50) / 50)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"momentum": mom, "rsi": rsi, "macd": macd}, regime)


class BreakoutStrategy(Strategy):
    """Buy securities breaking out of consolidation ranges."""
    def __init__(self):
        super().__init__(
            name="breakout",
            description="Buys securities breaking above resistance/Bollinger upper band",
            applicable_regimes=["STRONG_BULL", "BULL", "TRANSITION_BULL", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []

        bb_pos = market_data.get("bollinger_position", 0.5)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        adx = tech.get("trend_strength", 20)

        # Breakout signal: near upper Bollinger + high volume + rising ADX
        if bb_pos > 0.9:
            score += 15
            reasoning.append("At upper Bollinger Band — breakout potential")
        elif bb_pos > 0.7:
            score += 8
            reasoning.append("Approaching upper Bollinger Band")

        if volume_ratio > 1.5:
            score += 10
            reasoning.append(f"High volume confirmation ({volume_ratio:.1f}x average)")
        elif volume_ratio > 1.2:
            score += 5

        if adx > 25:
            score += 8
            reasoning.append("ADX rising — trend strength supporting breakout")
        elif adx < 15:
            score -= 10
            reasoning.append("ADX too low — no trend strength for breakout")

        # Penalize if already overbought
        rsi = tech.get("rsi", 50)
        if rsi > 80:
            score -= 10
            reasoning.append("RSI extremely overbought — breakout may be exhausted")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.75, (bb_pos * 0.4 + volume_ratio / 3 * 0.3 + adx / 50 * 0.3))

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"bb_position": bb_pos, "volume_ratio": volume_ratio, "adx": adx}, regime)


class EarningsMomentumStrategy(Strategy):
    """Buy securities with strong earnings beats and positive guidance."""
    def __init__(self):
        super().__init__(
            name="earnings_momentum",
            description="Buys after earnings beats with positive guidance and sentiment",
            applicable_regimes=["STRONG_BULL", "BULL", "NEUTRAL", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        fund = scores.get("fundamental", {})
        sent = scores.get("sentiment", {})

        score = 50.0
        reasoning = []

        earnings_growth = fund.get("earnings_growth", 50)
        revenue_growth = fund.get("revenue_growth", 50)
        transcript_sent = sent.get("transcript_sentiment", 50)
        news_sent = sent.get("news_sentiment", 50)

        score += (earnings_growth - 50) * 0.35
        score += (revenue_growth - 50) * 0.25
        score += (transcript_sent - 50) * 0.2
        score += (news_sent - 50) * 0.2

        if earnings_growth > 70:
            reasoning.append(f"Strong earnings growth ({earnings_growth:.0f})")
        if revenue_growth > 65:
            reasoning.append(f"Strong revenue growth ({revenue_growth:.0f})")
        if transcript_sent > 60:
            reasoning.append("Positive earnings call tone — management confident")

        reasoning.append(f"News sentiment: {news_sent:.0f}")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.7, (earnings_growth / 100) * 0.5 + (transcript_sent / 100) * 0.3 + (news_sent / 100) * 0.2)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"earnings_growth": earnings_growth, "revenue_growth": revenue_growth,
                                  "transcript_sentiment": transcript_sent, "news_sentiment": news_sent}, regime)


# ============================================================================
# MEAN REVERSION STRATEGIES
# ============================================================================

class MeanReversionStrategy(Strategy):
    """Buy oversold securities that are expected to revert to mean."""
    def __init__(self):
        super().__init__(
            name="mean_reversion",
            description="Buys oversold securities with positive fundamentals for mean reversion",
            applicable_regimes=["NEUTRAL", "TRANSITION_BULL"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        rsi = tech.get("rsi", 50)
        bb_pos = market_data.get("bollinger_position", 0.5)

        # Oversold conditions
        if rsi < 30:
            score += 20
            reasoning.append(f"RSI oversold ({rsi:.0f}) — prime reversion setup")
        elif rsi < 40:
            score += 10
            reasoning.append(f"RSI approaching oversold ({rsi:.0f})")
        elif rsi > 60:
            score -= 10
            reasoning.append("RSI not oversold — no reversion setup")

        if bb_pos < 0.2:
            score += 12
            reasoning.append("At lower Bollinger Band — stretched to downside")
        elif bb_pos < 0.3:
            score += 5

        # Only mean revert if fundamentals are solid (don't catch falling knives)
        roe = fund.get("roe", 50)
        if roe > 60:
            score += 8
            reasoning.append("Solid fundamentals — not a falling knife")
        elif roe < 30:
            score -= 10
            reasoning.append("Weak fundamentals — may be a value trap")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.7, (1 - rsi / 100) * 0.5 + roe / 100 * 0.5)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"rsi": rsi, "bb_position": bb_pos, "roe": roe}, regime)


class PairsTradingStrategy(Strategy):
    """Statistical arbitrage between correlated securities."""
    def __init__(self):
        super().__init__(
            name="pairs_trading",
            description="Long/short pairs based on spread mean reversion",
            applicable_regimes=["NEUTRAL", "TRANSITION_BULL", "TRANSITION_BEAR"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []

        rs = tech.get("relative_strength", 50)
        rsi = tech.get("rsi", 50)

        # In pairs trading, we go long the underperformer and short the outperformer
        # This strategy generates a NEUTRAL signal (pair would be constructed in portfolio)
        if rs < 45:
            score += 10
            reasoning.append("Underperforming benchmark — potential long leg of pair")
        elif rs > 55:
            score -= 10
            reasoning.append("Outperforming benchmark — potential short leg of pair")

        reasoning.append("Pairs signal requires correlated partner for execution")
        reasoning.append("Full pair construction happens in portfolio optimizer")

        # This strategy is market-neutral
        direction = SignalDirection.NEUTRAL
        confidence = 0.5

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"relative_strength": rs, "rsi": rsi}, regime)


class RSIReversalStrategy(Strategy):
    """RSI-based reversal — buy extreme oversold, sell extreme overbought."""
    def __init__(self):
        super().__init__(
            name="rsi_reversal",
            description="Reversal at RSI extremes (< 25 oversold, > 75 overbought)",
            applicable_regimes=["NEUTRAL", "TRANSITION_BULL", "TRANSITION_BEAR"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []
        direction = SignalDirection.NEUTRAL

        rsi = tech.get("rsi", 50)
        cci = tech.get("cci", 0) if "cci" in tech else 50
        williams = tech.get("williams_r", 50) if "williams_r" in tech else 50

        if rsi < 25:
            score = 75
            direction = SignalDirection.LONG
            reasoning.append(f"Extreme oversold RSI ({rsi:.0f}) — high probability bounce")
        elif rsi > 75:
            score = 25
            direction = SignalDirection.SHORT
            reasoning.append(f"Extreme overbought RSI ({rsi:.0f}) — high probability pullback")
        elif rsi < 35:
            score = 60
            direction = SignalDirection.LONG
            reasoning.append(f"Oversold RSI ({rsi:.0f}) — potential reversal")
        elif rsi > 65:
            score = 40
            direction = SignalDirection.NEUTRAL
            reasoning.append(f"Approaching overbought RSI ({rsi:.0f}) — watch for reversal")
        else:
            reasoning.append(f"RSI neutral ({rsi:.0f}) — no reversal signal")

        confidence = min(0.8, abs(rsi - 50) / 50)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"rsi": rsi, "cci": cci, "williams_r": williams}, regime)


# ============================================================================
# VALUE STRATEGIES
# ============================================================================

class ValueStrategy(Strategy):
    """Buy undervalued securities with strong fundamentals."""
    def __init__(self):
        super().__init__(
            name="value",
            description="Buys securities with low P/E, low P/B, strong balance sheets",
            applicable_regimes=["BULL", "TRANSITION_BULL", "NEUTRAL", "TRANSITION_BEAR", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        valuation = fund.get("valuation", 50)
        roe = fund.get("roe", 50)
        leverage = fund.get("leverage", 50)

        score += (valuation - 50) * 0.4  # Low P/E = high valuation score
        score += (roe - 50) * 0.3
        score += (leverage - 50) * 0.15

        if valuation > 70:
            reasoning.append(f"Attractive valuation (P/E score={valuation:.0f})")
        elif valuation < 30:
            reasoning.append(f"Expensive valuation (P/E score={valuation:.0f})")
            score -= 5

        if roe > 65:
            reasoning.append(f"Strong ROE ({roe:.0f})")
        if leverage > 65:
            reasoning.append("Low leverage — safe balance sheet")
        elif leverage < 35:
            reasoning.append("High leverage — value trap risk")
            score -= 8

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.8, (valuation / 100) * 0.4 + (roe / 100) * 0.4 + (leverage / 100) * 0.2)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"valuation": valuation, "roe": roe, "leverage": leverage}, regime)


class QualityStrategy(Strategy):
    """Buy high-quality companies with consistent profitability."""
    def __init__(self):
        super().__init__(
            name="quality",
            description="Buys high-ROE, high-margin, low-debt companies",
            applicable_regimes=["BULL", "TRANSITION_BULL", "NEUTRAL", "TRANSITION_BEAR", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        roe = fund.get("roe", 50)
        net_margin = fund.get("net_margin", 50)
        leverage = fund.get("leverage", 50)

        score += (roe - 50) * 0.4
        score += (net_margin - 50) * 0.3
        score += (leverage - 50) * 0.2

        if roe > 70:
            reasoning.append(f"Excellent ROE ({roe:.0f})")
        if net_margin > 65:
            reasoning.append(f"Strong net margin ({net_margin:.0f})")
        if leverage > 65:
            reasoning.append("Conservative balance sheet")

        # Quality compounds over time
        reasoning.append("Quality strategy: hold for long-term compounding")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.85, (roe / 100) * 0.5 + (net_margin / 100) * 0.3 + (leverage / 100) * 0.2)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"roe": roe, "net_margin": net_margin, "leverage": leverage}, regime)


class GrowthStrategy(Strategy):
    """Buy companies with strong revenue and earnings growth."""
    def __init__(self):
        super().__init__(
            name="growth",
            description="Buys companies with high revenue and earnings growth rates",
            applicable_regimes=["STRONG_BULL", "BULL", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        rev_growth = fund.get("revenue_growth", 50)
        earnings_growth = fund.get("earnings_growth", 50)
        valuation = fund.get("valuation", 50)

        score += (rev_growth - 50) * 0.35
        score += (earnings_growth - 50) * 0.35
        score += (valuation - 50) * 0.1  # Growth investors tolerate higher valuations

        # Growth at a reasonable price: penalize extreme valuations
        if valuation < 25:
            score -= 10
            reasoning.append("Very high P/E — growth priced to perfection")
        elif valuation > 50:
            score += 5
            reasoning.append("Reasonable valuation for growth rate")

        if rev_growth > 70:
            reasoning.append(f"Strong revenue growth ({rev_growth:.0f})")
        if earnings_growth > 70:
            reasoning.append(f"Strong earnings growth ({earnings_growth:.0f})")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.75, (rev_growth / 100) * 0.5 + (earnings_growth / 100) * 0.5)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"revenue_growth": rev_growth, "earnings_growth": earnings_growth,
                                  "valuation": valuation}, regime)


# ============================================================================
# DEFENSIVE STRATEGIES
# ============================================================================

class LowVolatilityStrategy(Strategy):
    """Buy low-volatility securities for downside protection."""
    def __init__(self):
        super().__init__(
            name="low_volatility",
            description="Buys low-volatility securities with stable returns",
            applicable_regimes=["TRANSITION_BEAR", "BEAR", "CRISIS"],
            config=StrategyConfig(max_position_pct=8.0, stop_loss_pct=7.0, take_profit_pct=10.0),
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        vol_score = tech.get("volatility", 50)
        leverage = fund.get("leverage", 50)
        net_margin = fund.get("net_margin", 50)

        # Low volatility = high score
        score += (vol_score - 50) * 0.4
        score += (leverage - 50) * 0.25
        score += (net_margin - 50) * 0.15

        if vol_score > 70:
            reasoning.append("Low volatility — stable price action")
        elif vol_score < 30:
            reasoning.append("High volatility — not suitable for defensive strategy")
            score -= 10

        if leverage > 65:
            reasoning.append("Low leverage — defensive balance sheet")

        reasoning.append(f"Defensive play in {regime} regime")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.8, (vol_score / 100) * 0.5 + (leverage / 100) * 0.3 + (net_margin / 100) * 0.2)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"volatility": vol_score, "leverage": leverage, "net_margin": net_margin}, regime)


class DefensiveStrategy(Strategy):
    """Capital preservation — rotate to cash-like or minimum-risk positions."""
    def __init__(self):
        super().__init__(
            name="defensive",
            description="Capital preservation — minimal risk, high-quality only",
            applicable_regimes=["BEAR", "CRISIS", "TRANSITION_BEAR"],
            config=StrategyConfig(max_position_pct=3.0, stop_loss_pct=4.0, take_profit_pct=5.0),
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        fund = scores.get("fundamental", {})
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []

        # In defensive mode, be extremely selective
        leverage = fund.get("leverage", 50)
        roe = fund.get("roe", 50)
        vol_score = tech.get("volatility", 50)

        # Only buy if all defensive criteria met
        if leverage > 70 and roe > 60 and vol_score > 65:
            score = 65
            reasoning.append("Meets defensive criteria: low debt, high ROE, low volatility")
        else:
            score = 35
            reasoning.append("Does not meet defensive criteria — avoid")

        if regime == "CRISIS":
            score -= 10
            reasoning.append("CRISIS regime — capital preservation paramount, minimal exposure")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = 0.9 if score > 60 else 0.3  # High confidence when defensive criteria met

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"leverage": leverage, "roe": roe, "volatility": vol_score}, regime)


class OptionsCollarStrategy(Strategy):
    """Protective collar — buy stock + buy protective put + sell call."""
    def __init__(self):
        super().__init__(
            name="options_collar",
            description="Protective collar for downside protection in bearish regimes",
            applicable_regimes=["BEAR", "CRISIS", "TRANSITION_BEAR"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        options = scores.get("options", {})
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        iv = options.get("implied_volatility", 50)
        put_call = options.get("put_call_ratio", 50)

        # Collar is attractive when IV is moderate (not too expensive to buy puts)
        if 40 < iv < 65:
            score += 10
            reasoning.append("Moderate IV — puts are affordable for hedging")
        elif iv > 80:
            score -= 10
            reasoning.append("High IV — put protection too expensive")

        if put_call > 60:
            reasoning.append("Elevated put/call ratio confirms hedging need")
            score += 5

        # Still want quality underlying
        roe = fund.get("roe", 50)
        if roe > 60:
            score += 8
            reasoning.append("Quality underlying — worth protecting")

        reasoning.append("Collar: long stock + long put + short call (zero-cost or low-cost)")
        reasoning.append("Caps upside but protects downside — for existing positions")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = 0.7

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"iv": iv, "put_call": put_call, "roe": roe}, regime)


# ============================================================================
# MACRO STRATEGIES
# ============================================================================

class SectorRotationStrategy(Strategy):
    """Rotate into sectors that benefit from the current macro regime."""
    def __init__(self):
        super().__init__(
            name="sector_rotation",
            description="Rotates sectors based on MCEI/regime — cyclical in bull, defensive in bear",
            applicable_regimes=["STRONG_BULL", "BULL", "TRANSITION_BULL", "TRANSITION_BEAR", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        macro = scores.get("macro", {})
        tech = scores.get("technical", {})

        score = 50.0
        reasoning = []

        macro_alignment = macro.get("macro_alignment", 50)
        rs = tech.get("relative_strength", 50)

        score += (macro_alignment - 50) * 0.4
        score += (rs - 50) * 0.3

        if macro_alignment > 65 and rs > 55:
            reasoning.append("Sector benefits from current macro environment + outperforming")
            score += 10
        elif macro_alignment < 40:
            reasoning.append("Sector misaligned with macro regime — rotate away")
            score -= 10

        reasoning.append(f"Macro alignment: {macro_alignment:.0f}")
        reasoning.append(f"Relative strength: {rs:.0f}")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.75, (macro_alignment / 100) * 0.5 + (rs / 100) * 0.5)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"macro_alignment": macro_alignment, "relative_strength": rs}, regime)


class MacroDrivenStrategy(Strategy):
    """Trade based on MCEI and liquidity conditions — macro-first approach."""
    def __init__(self):
        super().__init__(
            name="macro_driven",
            description="MCEI-driven strategy — liquidity expansion = long, contraction = defensive",
            applicable_regimes=["STRONG_BULL", "BULL", "NEUTRAL", "TRANSITION_BEAR", "BEAR", "CRISIS", "RECOVERY"],
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        macro = scores.get("macro", {})

        score = 50.0
        reasoning = []

        mcei_score = macro.get("macro_alignment", 50)
        regime_score = macro.get("mcei_regime_score", 50)
        regime_conf = macro.get("regime_confidence", 50)

        score += (mcei_score - 50) * 0.4
        score += (regime_score - 50) * 0.3
        score += (regime_conf - 50) * 0.1

        if mcei_score > 65:
            reasoning.append(f"MCEI expansionary ({mcei_score:.0f}) — risk-on")
            direction = SignalDirection.LONG
        elif mcei_score < 35:
            reasoning.append(f"MCEI contractionary ({mcei_score:.0f}) — risk-off")
            direction = SignalDirection.SHORT if score < 30 else SignalDirection.NEUTRAL
            if direction == SignalDirection.SHORT:
                reasoning.append("Short candidate — macro headwinds")
        else:
            reasoning.append(f"MCEI neutral ({mcei_score:.0f}) — no strong macro signal")
            direction = SignalDirection.NEUTRAL

        reasoning.append(f"Regime confidence: {regime_conf:.0f}%")

        confidence = min(0.8, abs(mcei_score - 50) / 50 * regime_conf / 100)

        return self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                 {"mcei": mcei_score, "regime_score": regime_score, "regime_confidence": regime_conf}, regime)


# ============================================================================
# SIZING STRATEGY
# ============================================================================

class RiskParityStrategy(Strategy):
    """Equal risk contribution — size positions by inverse volatility."""
    def __init__(self):
        super().__init__(
            name="risk_parity",
            description="Equal risk contribution — lower volatility = larger position",
            applicable_regimes=["STRONG_BULL", "BULL", "NEUTRAL", "TRANSITION_BEAR", "BEAR", "CRISIS", "RECOVERY"],
            config=StrategyConfig(max_position_pct=6.0, stop_loss_pct=6.0, take_profit_pct=12.0),
        )

    def evaluate(self, symbol, evidence, debate, market_data, regime):
        scores = self._get_agent_scores(evidence)
        tech = scores.get("technical", {})
        fund = scores.get("fundamental", {})

        score = 50.0
        reasoning = []

        vol_score = tech.get("volatility", 50)
        trend_score = tech.get("trend", 50)
        roe = fund.get("roe", 50)

        # Risk parity: lower volatility = higher allocation
        score += (vol_score - 50) * 0.5  # Low vol = high score
        score += (trend_score - 50) * 0.2
        score += (roe - 50) * 0.15

        if vol_score > 70:
            reasoning.append("Low volatility — larger risk parity allocation")
        elif vol_score < 30:
            reasoning.append("High volatility — smaller risk parity allocation")
            score -= 5

        reasoning.append("Risk parity: position size inversely proportional to volatility")
        reasoning.append("All positions contribute equal risk to portfolio")

        direction = SignalDirection.LONG if score >= self.config.min_score else SignalDirection.NEUTRAL
        confidence = min(0.75, vol_score / 100)

        # Override position size: inverse volatility weighting
        signal = self._make_signal(symbol, direction, score, confidence, market_data, reasoning,
                                   {"volatility": vol_score, "trend": trend_score, "roe": roe}, regime)
        if direction == SignalDirection.LONG:
            # Inverse vol sizing: low vol gets more, high vol gets less
            vol_pct = market_data.get("atr", 0) / market_data.get("price", 1) * 100 if market_data.get("price") else 2
            vol_pct = max(0.5, min(10.0, vol_pct))
            signal.position_size_pct = self.config.max_position_pct * (2.0 / vol_pct)
            signal.position_size_pct = min(self.config.max_position_pct, signal.position_size_pct)

        return signal


# ============================================================================
# STRATEGY FACTORY
# ============================================================================

def create_all_strategies() -> list[Strategy]:
    """Create instances of all 16 strategies."""
    return [
        TrendFollowingStrategy(),
        MomentumStrategy(),
        BreakoutStrategy(),
        EarningsMomentumStrategy(),
        MeanReversionStrategy(),
        PairsTradingStrategy(),
        RSIReversalStrategy(),
        ValueStrategy(),
        QualityStrategy(),
        GrowthStrategy(),
        LowVolatilityStrategy(),
        DefensiveStrategy(),
        OptionsCollarStrategy(),
        SectorRotationStrategy(),
        MacroDrivenStrategy(),
        RiskParityStrategy(),
    ]


def get_strategies_for_regime(regime: str) -> list[Strategy]:
    """Get all strategies applicable to a given regime."""
    return [s for s in create_all_strategies() if s.is_applicable(regime)]
