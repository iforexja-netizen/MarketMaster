"""
Phase 4 tests — Strategies, screener, backtester, walk-forward, portfolio optimizer,
and trade construction.

Tests computation and analysis logic without requiring a database connection.
"""

import pytest
from datetime import date, datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


# ============================================================================
# Mock Data (reuse from Phase 3)
# ============================================================================

@dataclass
class MockBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class MockEvidence:
    agent: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    observations: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    bull_case: list = field(default_factory=list)
    bear_case: list = field(default_factory=list)
    risks: list = field(default_factory=list)
    data_quality: float = 0.8
    confidence: float = 0.7
    recommended_actions: list = field(default_factory=list)


@dataclass
class MockDebate:
    bull_score: float = 65.0
    bear_score: float = 35.0
    net_score: float = 30.0
    winner: str = "bull"
    confidence: float = 0.7
    contradictions: list = field(default_factory=list)
    key_risks: list = field(default_factory=list)
    summary: str = "Bull case wins"


def make_price_bars(n=300, start_price=150, seed=42):
    """Generate synthetic OHLCV bars for backtesting."""
    np.random.seed(seed)
    # Slight uptrend with noise
    daily_returns = np.random.normal(0.0005, 0.02, n)
    prices = start_price * np.cumprod(1 + daily_returns)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date

    bars = []
    for i in range(n):
        p = float(prices[i])
        bars.append(MockBar(
            date=dates[i],
            open=p * 0.998,
            high=p * 1.015,
            low=p * 0.985,
            close=p,
            volume=int(1_000_000 + np.random.randint(-200_000, 200_000)),
        ))
    return bars


def make_bullish_evidence():
    """Evidence that should trigger bullish strategies."""
    return [
        MockEvidence(
            agent="macro",
            scores={"macro_alignment": 70, "mcei_regime_score": 75, "regime_confidence": 80},
            bull_case=["Macro expansionary"],
            confidence=0.8, data_quality=0.9,
        ),
        MockEvidence(
            agent="technical",
            scores={"trend": 75, "trend_strength": 30, "rsi": 62, "macd_momentum": 65,
                    "momentum": 65, "relative_strength": 60, "volatility": 65, "rsi": 58},
            bull_case=["Strong uptrend", "RSI in momentum zone"],
            confidence=0.75, data_quality=1.0,
        ),
        MockEvidence(
            agent="fundamental",
            scores={"roe": 75, "net_margin": 70, "valuation": 65, "leverage": 70,
                    "revenue_growth": 68, "earnings_growth": 72},
            bull_case=["Strong ROE", "Growing revenue"],
            confidence=0.7, data_quality=0.8,
        ),
    ]


def make_bearish_evidence():
    """Evidence that should trigger bearish/defensive strategies."""
    return [
        MockEvidence(
            agent="macro",
            scores={"macro_alignment": 25, "mcei_regime_score": 20, "regime_confidence": 75},
            bear_case=["Macro contractionary"],
            confidence=0.8, data_quality=0.9,
        ),
        MockEvidence(
            agent="technical",
            scores={"trend": 25, "trend_strength": 35, "rsi": 28, "volatility": 30,
                    "relative_strength": 35},
            bear_case=["Downtrend", "RSI oversold"],
            confidence=0.7, data_quality=1.0,
        ),
        MockEvidence(
            agent="fundamental",
            scores={"roe": 25, "net_margin": 20, "valuation": 30, "leverage": 25,
                    "revenue_growth": 20, "earnings_growth": 15},
            bear_case=["Weak fundamentals"],
            confidence=0.65, data_quality=0.7,
        ),
    ]


def make_market_data(price=150.0):
    return {
        "price": price,
        "atr": price * 0.02,
        "sma20": price * 0.98,
        "sma50": price * 0.95,
        "sma200": price * 0.90,
        "volume_ratio": 1.2,
        "bollinger_position": 0.6,
    }


# ============================================================================
# Strategy Registry Tests
# ============================================================================

class TestStrategyRegistry:
    def test_registry_has_16_strategies(self):
        from marketmaster.strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        assert registry.count() == 16

    def test_all_strategies_have_names(self):
        from marketmaster.strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        for s in registry.all():
            assert s.name is not None
            assert s.description is not None
            assert len(s.applicable_regimes) > 0

    def test_regime_filter(self):
        from marketmaster.strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        bull_strategies = registry.for_regime("BULL")
        assert len(bull_strategies) > 0
        for s in bull_strategies:
            assert "BULL" in s.applicable_regimes

    def test_crisis_has_defensive_strategies(self):
        from marketmaster.strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        crisis_strategies = registry.for_regime("CRISIS")
        names = [s.name for s in crisis_strategies]
        assert "defensive" in names
        assert "risk_parity" in names

    def test_get_by_name(self):
        from marketmaster.strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        s = registry.get("trend_following")
        assert s.name == "trend_following"


# ============================================================================
# Strategy Tests
# ============================================================================

class TestStrategies:
    def test_trend_following_bullish(self):
        from marketmaster.strategies.strategies import TrendFollowingStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = TrendFollowingStrategy()
        signal = strategy.evaluate("AAPL", make_bullish_evidence(), MockDebate(), make_market_data(), "BULL")

        assert signal.strategy_name == "trend_following"
        assert signal.direction == SignalDirection.LONG
        assert signal.score > 50
        assert signal.entry_price is not None
        assert signal.stop_price is not None
        assert signal.target_price is not None
        assert signal.risk_reward_ratio > 0
        assert len(signal.reasoning) > 0

    def test_trend_following_bearish(self):
        from marketmaster.strategies.strategies import TrendFollowingStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = TrendFollowingStrategy()
        signal = bearish_market = make_market_data(140)
        bearish_market['sma200'] = 160  # Price below SMA200 = downtrend
        bearish_market['sma50'] = 150
        signal = strategy.evaluate("AAPL", make_bearish_evidence(), MockDebate(bull_score=30, bear_score=70, net_score=-40), bearish_market, "BEAR")

        # Should not trigger in bearish regime (not applicable)
        assert signal.direction == SignalDirection.NEUTRAL or signal.score < 60

    def test_mean_reversion_oversold(self):
        from marketmaster.strategies.strategies import MeanReversionStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = MeanReversionStrategy()

        # Create evidence with low RSI
        evidence = make_bullish_evidence()
        evidence[1].scores["rsi"] = 25  # Oversold
        market_data = make_market_data()
        market_data["bollinger_position"] = 0.15  # Near lower band

        signal = strategy.evaluate("AAPL", evidence, MockDebate(), market_data, "NEUTRAL")
        assert signal.score > 60
        assert signal.direction == SignalDirection.LONG

    def test_value_strategy(self):
        from marketmaster.strategies.strategies import ValueStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = ValueStrategy()
        signal = strategy.evaluate("AAPL", make_bullish_evidence(), MockDebate(), make_market_data(), "BULL")

        assert signal.strategy_name == "value"
        assert signal.score > 55
        assert signal.direction == SignalDirection.LONG

    def test_defensive_strategy_in_crisis(self):
        from marketmaster.strategies.strategies import DefensiveStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = DefensiveStrategy()

        # In crisis, should only recommend highest quality
        evidence = make_bullish_evidence()
        signal = strategy.evaluate("AAPL", evidence, MockDebate(), make_market_data(120), "CRISIS")
        assert signal.score < 60  # Should be very selective
        assert signal.direction == SignalDirection.NEUTRAL or signal.direction == SignalDirection.LONG

    def test_macro_driven_strategy_bullish(self):
        from marketmaster.strategies.strategies import MacroDrivenStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = MacroDrivenStrategy()
        signal = strategy.evaluate("AAPL", make_bullish_evidence(), MockDebate(), make_market_data(), "BULL")

        assert signal.direction == SignalDirection.LONG
        assert signal.score > 55

    def test_macro_driven_strategy_bearish(self):
        from marketmaster.strategies.strategies import MacroDrivenStrategy
        from marketmaster.strategies.base import SignalDirection
        strategy = MacroDrivenStrategy()
        signal = bearish_market = make_market_data(140)
        bearish_market['sma200'] = 160  # Price below SMA200 = downtrend
        bearish_market['sma50'] = 150
        signal = strategy.evaluate("AAPL", make_bearish_evidence(), MockDebate(bull_score=30, bear_score=70, net_score=-40), bearish_market, "BEAR")

        # Should be short or neutral in bearish macro
        assert signal.direction in (SignalDirection.SHORT, SignalDirection.NEUTRAL)

    def test_all_16_strategies_evaluate_without_error(self):
        from marketmaster.strategies.strategies import create_all_strategies
        strategies = create_all_strategies()
        assert len(strategies) == 16

        for strategy in strategies:
            signal = strategy.evaluate("AAPL", make_bullish_evidence(), MockDebate(), make_market_data(), "BULL")
            assert signal is not None
            assert signal.strategy_name == strategy.name
            assert 0 <= signal.score <= 100
            assert 0 <= signal.confidence <= 1

    def test_position_size_scales_with_score(self):
        from marketmaster.strategies.strategies import TrendFollowingStrategy
        strategy = TrendFollowingStrategy()

        # High score → larger position
        signal_high = strategy.evaluate("AAPL", make_bullish_evidence(), MockDebate(), make_market_data(), "BULL")

        # Lower score evidence
        weak_evidence = make_bullish_evidence()
        for ev in weak_evidence:
            for k in ev.scores:
                ev.scores[k] = 52
        signal_low = strategy.evaluate("AAPL", weak_evidence, MockDebate(), make_market_data(), "BULL")

        if signal_high.direction.value == "long" and signal_low.direction.value == "long":
            assert signal_high.position_size_pct >= signal_low.position_size_pct


# ============================================================================
# Screener Tests
# ============================================================================

class TestScreener:
    def test_screener_creation(self):
        from marketmaster.strategies.screener import Screener
        screener = Screener()
        assert screener is not None

    def test_screen_with_mock_analysis(self):
        from marketmaster.strategies.screener import Screener
        from marketmaster.agents.orchestrator import AnalysisResult

        # Create a mock analyze function
        def mock_analyze(symbol, as_of):
            result = AnalysisResult(symbol=symbol, as_of=as_of, data_available=True)
            result.evidence = make_bullish_evidence()
            result.debate = MockDebate()
            result.notes = [f"Latest price: $150.00 on {as_of}"]
            return result

        screener = Screener()
        result = screener.scan(
            universe=["AAPL", "MSFT", "GOOGL"],
            regime="BULL",
            as_of=date(2025, 6, 1),
            analyze_fn=mock_analyze,
            top_n=10,
        )

        assert result.regime == "BULL"
        assert result.screened == 3
        assert len(result.active_strategies) > 0
        assert len(result.signals) > 0  # Should find some signals

    def test_screen_filters_low_scores(self):
        from marketmaster.strategies.screener import Screener
        from marketmaster.agents.orchestrator import AnalysisResult

        def mock_analyze(symbol, as_of):
            result = AnalysisResult(symbol=symbol, as_of=as_of, data_available=True)
            result.evidence = make_bearish_evidence()  # Low scores
            result.debate = MockDebate(bull_score=30, bear_score=70, net_score=-40)
            result.notes = [f"Latest price: $100.00 on {as_of}"]
            return result

        screener = Screener()
        result = screener.scan(
            universe=["AAPL"],
            regime="BULL",
            as_of=date(2025, 6, 1),
            analyze_fn=mock_analyze,
            min_score=70,
        )

        # Should have fewer or no signals with bearish evidence + high threshold
        for s in result.signals:
            assert s.score >= 70


# ============================================================================
# Backtester Tests
# ============================================================================

class TestBacktester:
    def test_backtest_runs(self):
        from marketmaster.backtest.engine import BacktestEngine
        from marketmaster.strategies.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy()
        bars = make_price_bars(300)
        engine = BacktestEngine(initial_capital=100_000)

        result = engine.run(
            strategy=strategy,
            symbol="AAPL",
            bars=bars,
            regime="BULL",
        )

        assert result.symbol == "AAPL"
        assert result.strategy_name == "trend_following"
        assert result.initial_capital == 100_000
        assert result.final_equity > 0
        assert len(result.equity_curve) == 300
        assert result.metrics.total_trades >= 0

    def test_backtest_empty_bars(self):
        from marketmaster.backtest.engine import BacktestEngine
        from marketmaster.strategies.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy()
        engine = BacktestEngine()

        result = engine.run(
            strategy=strategy,
            symbol="AAPL",
            bars=[],
            regime="BULL",
        )

        assert result.final_equity == result.initial_capital
        assert "No bars" in result.notes[0]

    def test_backtest_metrics_computed(self):
        from marketmaster.backtest.engine import BacktestEngine
        from marketmaster.strategies.strategies import MomentumStrategy

        strategy = MomentumStrategy()
        bars = make_price_bars(250, seed=123)
        engine = BacktestEngine()

        result = engine.run(
            strategy=strategy,
            symbol="AAPL",
            bars=bars,
            regime="BULL",
        )

        m = result.metrics
        assert m.total_return_pct is not None
        assert m.max_drawdown_pct >= 0
        assert m.total_trades >= 0
        assert m.win_rate >= 0

    def test_backtest_stop_loss_triggers(self):
        from marketmaster.backtest.engine import BacktestEngine
        from marketmaster.strategies.base import Strategy, SignalDirection

        # Create a strategy that always enters long at market
        class AlwaysLongStrategy(Strategy):
            def __init__(self):
                super().__init__("test_always_long", "test", ["BULL"],
                                 __import__('marketmaster.strategies.base', fromlist=['StrategyConfig']).StrategyConfig(
                                     stop_loss_pct=3.0, take_profit_pct=50.0))

            def evaluate(self, symbol, evidence, debate, market_data, regime):
                return self._make_signal(symbol, SignalDirection.LONG, 80, 0.8, market_data,
                                        ["Always long"], {}, regime)

        # Create bars that will trigger stop loss
        bars = []
        dates = pd.date_range("2024-01-01", periods=20, freq="B").date
        for i in range(20):
            if i < 5:
                price = 100.0  # Entry around 100
            else:
                price = 94.0  # Drop below 3% stop
            bars.append(MockBar(date=dates[i], open=price, high=price+0.5, low=price-0.5, close=price, volume=1000000))

        engine = BacktestEngine()
        result = engine.run(
            strategy=AlwaysLongStrategy(),
            symbol="TEST",
            bars=bars,
            regime="BULL",
        )

        # Should have at least one trade (entry + stop exit)
        assert result.metrics.total_trades >= 1

    def test_backtest_with_date_filter(self):
        from marketmaster.backtest.engine import BacktestEngine
        from marketmaster.strategies.strategies import ValueStrategy

        strategy = ValueStrategy()
        bars = make_price_bars(300)
        engine = BacktestEngine()

        result = engine.run(
            strategy=strategy,
            symbol="AAPL",
            bars=bars,
            regime="NEUTRAL",
            start_date=bars[50].date,
            end_date=bars[250].date,
        )

        assert result.start_date >= bars[50].date
        assert result.end_date <= bars[250].date


# ============================================================================
# Walk-Forward Tests
# ============================================================================

class TestWalkForward:
    def test_walkforward_runs(self):
        from marketmaster.backtest.walk_forward import WalkForwardValidator
        from marketmaster.strategies.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy()
        bars = make_price_bars(800)  # Enough for multiple windows
        validator = WalkForwardValidator(
            train_days=200,
            test_days=50,
            step_days=50,
        )

        result = validator.run(
            strategy=strategy,
            symbol="AAPL",
            bars=bars,
            regime="BULL",
        )

        assert len(result.windows) > 0
        assert result.avg_is_return is not None
        assert result.avg_oos_return is not None
        assert result.wfe_ratio is not None
        assert len(result.summary) > 0

    def test_walkforward_insufficient_data(self):
        from marketmaster.backtest.walk_forward import WalkForwardValidator
        from marketmaster.strategies.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy()
        bars = make_price_bars(50)  # Too few
        validator = WalkForwardValidator(train_days=200, test_days=50, step_days=50)

        result = validator.run(strategy=strategy, symbol="AAPL", bars=bars, regime="BULL")

        assert len(result.windows) == 0
        assert "Insufficient" in result.summary

    def test_walkforward_overfit_detection(self):
        from marketmaster.backtest.walk_forward import WalkForwardValidator
        from marketmaster.strategies.strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy()
        bars = make_price_bars(800)
        validator = WalkForwardValidator(train_days=200, test_days=50, step_days=50)

        result = validator.run(strategy=strategy, symbol="AAPL", bars=bars, regime="BULL")

        # WFE should be computed
        assert result.wfe_ratio is not None
        # overfit_warning should be True if WFE < 0.25
        if result.wfe_ratio < 0.25:
            assert result.overfit_warning


# ============================================================================
# Portfolio Optimizer Tests
# ============================================================================

class TestPortfolioOptimizer:
    def _make_signals(self, n=5):
        from marketmaster.strategies.base import TradeSignal, SignalDirection
        return [
            TradeSignal(
                symbol=f"STK{i}",
                strategy_name="trend_following",
                direction=SignalDirection.LONG,
                score=60 + i * 5,
                confidence=0.5 + i * 0.05,
                entry_price=100.0 + i * 10,
                stop_price=95.0 + i * 10,
                target_price=115.0 + i * 10,
                position_size_pct=5.0,
                risk_reward_ratio=3.0,
            )
            for i in range(n)
        ]

    def test_equal_weight(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(initial_capital=100_000)
        signals = self._make_signals(5)

        allocation = opt.optimize(
            signals=signals,
            method="equal_weight",
            max_positions=10,
            regime="BULL",
        )

        assert allocation.n_positions == 5
        assert allocation.total_allocation > 0
        # Each position should have roughly equal weight
        weights = [p.weight for p in allocation.positions]
        assert max(weights) - min(weights) < 0.02  # Nearly equal

    def test_score_weighted(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(initial_capital=100_000)
        signals = self._make_signals(5)

        allocation = opt.optimize(
            signals=signals,
            method="score_weighted",
            max_positions=10,
            max_position_pct=20.0,  # High enough so weights aren't all capped
            regime="BULL",
        )

        assert allocation.n_positions == 5
        # Higher-scored signals should get larger weight
        weights = [p.weight for p in allocation.positions]
        assert weights[0] < weights[-1]  # First signal has lowest score

    def test_risk_parity(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(initial_capital=100_000)
        signals = self._make_signals(5)

        allocation = opt.optimize(
            signals=signals,
            method="risk_parity",
            max_positions=10,
            regime="BULL",
        )

        assert allocation.n_positions == 5
        assert allocation.total_allocation > 0

    def test_regime_reduces_exposure_in_bear(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        signals = self._make_signals(5)

        bull_alloc = opt.optimize(signals=signals, regime="STRONG_BULL")
        bear_alloc = opt.optimize(signals=signals, regime="CRISIS")

        assert bull_alloc.total_allocation > bear_alloc.total_allocation

    def test_max_positions_limit(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        signals = self._make_signals(10)

        allocation = opt.optimize(
            signals=signals,
            max_positions=3,
            regime="BULL",
        )

        assert allocation.n_positions == 3

    def test_no_signals(self):
        from marketmaster.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        allocation = opt.optimize(signals=[], regime="BULL")
        assert allocation.n_positions == 0


# ============================================================================
# Trade Construction Tests
# ============================================================================

class TestTradeConstruction:
    def _make_allocation(self):
        from marketmaster.portfolio.optimizer import PortfolioAllocation, PositionAllocation
        return PortfolioAllocation(
            as_of=date(2025, 6, 1),
            total_allocation=0.75,
            cash_reserve=0.25,
            n_positions=2,
            positions=[
                PositionAllocation(
                    symbol="AAPL", strategy_name="trend_following", direction="long",
                    weight=0.05, dollar_allocation=5000, shares=50,
                    entry_price=100.0, stop_price=95.0, target_price=115.0,
                    risk_reward_ratio=3.0, score=75, confidence=0.7,
                ),
                PositionAllocation(
                    symbol="MSFT", strategy_name="value", direction="long",
                    weight=0.04, dollar_allocation=4000, shares=20,
                    entry_price=200.0, stop_price=190.0, target_price=230.0,
                    risk_reward_ratio=3.0, score=70, confidence=0.65,
                ),
            ],
        )

    def test_construct_orders(self):
        from marketmaster.portfolio.construction import TradeConstructor
        constructor = TradeConstructor()
        plan = constructor.construct(self._make_allocation())

        assert plan.n_positions == 2
        assert len(plan.orders) == 2  # 2 entry orders
        assert len(plan.stop_orders) == 2  # 2 stop orders
        assert len(plan.target_orders) == 2  # 2 target orders
        assert plan.total_risk_pct > 0

    def test_market_orders_when_no_limit(self):
        from marketmaster.portfolio.construction import TradeConstructor
        constructor = TradeConstructor(use_limit_orders=False)
        plan = constructor.construct(self._make_allocation())

        for order in plan.orders:
            assert order.order_type.value == "market"

    def test_limit_orders_by_default(self):
        from marketmaster.portfolio.construction import TradeConstructor
        constructor = TradeConstructor(use_limit_orders=True)
        plan = constructor.construct(self._make_allocation())

        for order in plan.orders:
            assert order.order_type.value == "limit"
            assert order.limit_price is not None

    def test_stop_orders_on_correct_side(self):
        from marketmaster.portfolio.construction import TradeConstructor
        constructor = TradeConstructor()
        plan = constructor.construct(self._make_allocation())

        for stop in plan.stop_orders:
            # Long positions have sell stops below entry
            assert stop.stop_price is not None
            assert stop.side.value == "sell"  # Long → sell to close

    def test_rejected_positions(self):
        from marketmaster.portfolio.construction import TradeConstructor
        from marketmaster.portfolio.optimizer import PortfolioAllocation, PositionAllocation

        alloc = PortfolioAllocation(
            as_of=date(2025, 6, 1),
            total_allocation=0.05,
            cash_reserve=0.95,
            n_positions=1,
            positions=[
                PositionAllocation(
                    symbol="BAD", strategy_name="test", direction="long",
                    weight=0.05, dollar_allocation=5000, shares=10,
                    entry_price=None,  # No price → should be rejected
                ),
            ],
        )

        constructor = TradeConstructor()
        plan = constructor.construct(alloc)

        assert len(plan.rejected) == 1
        assert "no entry price" in plan.rejected[0]
