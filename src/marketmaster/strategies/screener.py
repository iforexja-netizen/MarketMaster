"""
Screener — Scans the security universe to identify trade opportunities.

The screener:
1. Takes a universe of securities
2. For each security, gets agent evidence + debate result from the orchestrator
3. Runs all applicable strategies (based on current regime)
4. Collects all trade signals above threshold
5. Ranks by score × confidence and returns the top opportunities

This is the bridge between the research layer (agents + debate) and the
strategy layer (strategies → signals → portfolio).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Any

from marketmaster.strategies.base import Strategy, TradeSignal, SignalDirection
from marketmaster.strategies.registry import StrategyRegistry


@dataclass
class ScreeningResult:
    """Result of screening the universe."""
    as_of: date
    regime: str
    universe_size: int
    screened: int
    signals: list[TradeSignal] = field(default_factory=list)
    top_opportunities: list[TradeSignal] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    active_strategies: list[str] = field(default_factory=list)


class Screener:
    """
    Screens the security universe for trade opportunities.

    Usage:
        screener = Screener(registry)
        result = screener.scan(
            universe=["AAPL", "MSFT", "GOOGL", ...],
            regime="BULL",
            as_of=date(2025, 6, 1),
            analyze_fn=orchestrator.analyze,
        )
    """

    def __init__(self, registry: Optional[StrategyRegistry] = None):
        self.registry = registry or StrategyRegistry()

    def scan(
        self,
        universe: list[str],
        regime: str,
        as_of: date,
        analyze_fn: Any,  # orchestrator.analyze
        top_n: int = 20,
        min_score: float = 55.0,
        min_confidence: float = 0.25,
    ) -> ScreeningResult:
        """
        Screen the universe and return ranked trade signals.

        Args:
            universe: List of ticker symbols
            regime: Current market regime
            as_of: Point-in-time date
            analyze_fn: Function(symbol, as_of) → AnalysisResult
            top_n: Number of top opportunities to return
            min_score: Minimum signal score to include
            min_confidence: Minimum confidence to include

        Returns:
            ScreeningResult with all qualifying signals
        """
        active_strategies = self.registry.for_regime(regime)
        active_strategy_names = [s.name for s in active_strategies]

        result = ScreeningResult(
            as_of=as_of,
            regime=regime,
            universe_size=len(universe),
            screened=0,
            active_strategies=active_strategy_names,
        )

        for symbol in universe:
            try:
                # Get full analysis from orchestrator
                analysis = analyze_fn(symbol, as_of)

                if not analysis.data_available:
                    continue

                result.screened += 1

                # Extract evidence and debate from analysis
                evidence = analysis.evidence
                debate = analysis.debate if hasattr(analysis, 'debate') else None

                # Get latest price data for strategy evaluation
                market_data = self._extract_market_data(analysis, evidence)

                # Run all applicable strategies
                for strategy in active_strategies:
                    try:
                        signal = strategy.evaluate(
                            symbol=symbol,
                            evidence=evidence,
                            debate=debate,
                            market_data=market_data,
                            regime=regime,
                        )

                        # Filter: only keep actionable signals
                        if (signal.direction != SignalDirection.NEUTRAL
                            and signal.score >= min_score
                            and signal.confidence >= min_confidence):
                            signal.as_of = as_of
                            result.signals.append(signal)

                    except Exception as e:
                        result.errors.append(f"{symbol}/{strategy.name}: {e}")

            except Exception as e:
                result.errors.append(f"{symbol}: {e}")

        # Rank signals by score × confidence
        result.signals.sort(key=lambda s: s.score * s.confidence, reverse=True)
        result.top_opportunities = result.signals[:top_n]

        return result

    def _extract_market_data(self, analysis: Any, evidence: list[Any]) -> dict[str, float]:
        """Extract market data (price, ATR, etc.) from analysis and evidence."""
        market_data = {}

        # Get latest price from analysis
        if hasattr(analysis, 'notes'):
            for note in analysis.notes:
                if 'Latest price:' in note:
                    # Parse "Latest price: $191.00 on 2025-06-01"
                    try:
                        price_str = note.split('$')[1].split(' ')[0]
                        market_data["price"] = float(price_str)
                    except (IndexError, ValueError):
                        pass

        # Extract technical indicators from evidence
        for ev in evidence:
            if ev.agent == "technical":
                # Pull values from scores
                if "trend" in ev.scores:
                    market_data["trend_score"] = ev.scores["trend"]
                if "trend_strength" in ev.scores:
                    market_data["adx"] = ev.scores["trend_strength"]
                if "rsi" in ev.scores:
                    market_data["rsi"] = ev.scores["rsi"]
                if "relative_strength" in ev.scores:
                    market_data["relative_strength"] = ev.scores["relative_strength"]
                if "volatility" in ev.scores:
                    market_data["volatility_score"] = ev.scores["volatility"]

                # Look for ATR and volume in observations
                for obs in ev.observations:
                    if "ATR" in obs and "of price" in obs:
                        try:
                            atr_str = obs.split("$")[1].split(" ")[0]
                            market_data["atr"] = float(atr_str)
                        except (IndexError, ValueError):
                            pass
                    if "Volume ratio" in obs:
                        try:
                            vol_str = obs.split(":")[1].strip().split("x")[0]
                            market_data["volume_ratio"] = float(vol_str)
                        except (IndexError, ValueError):
                            pass
                    if "Bollinger position" in obs:
                        try:
                            pos_str = obs.split(":")[1].strip()
                            market_data["bollinger_position"] = float(pos_str)
                        except (IndexError, ValueError):
                            pass

            # Also get SMA values from observations
            if ev.agent == "technical":
                for obs in ev.observations:
                    if "SMA20" in obs and "$" in obs:
                        try:
                            sma_str = obs.split("SMA20: $")[1].split(",")[0].strip()
                            market_data["sma20"] = float(sma_str)
                        except (IndexError, ValueError):
                            pass

        # Default values if not extracted
        if "price" not in market_data:
            market_data["price"] = 100.0
        if "atr" not in market_data:
            market_data["atr"] = market_data["price"] * 0.02
        if "volume_ratio" not in market_data:
            market_data["volume_ratio"] = 1.0
        if "bollinger_position" not in market_data:
            market_data["bollinger_position"] = 0.5
        if "sma200" not in market_data:
            # Infer from trend score
            trend = market_data.get("trend_score", 50)
            market_data["sma200"] = market_data["price"] * (0.95 if trend > 50 else 1.05)

        return market_data
