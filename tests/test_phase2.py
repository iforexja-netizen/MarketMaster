"""
Phase 2 tests — MCEI pipeline, technical indicators, quant engine.

Tests the computation logic without requiring a database connection.
"""

import pytest
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd


class TestTechnicalIndicators:
    """Test the pure technical indicator functions."""

    def _make_price_data(self, n=250, seed=42):
        """Generate synthetic OHLCV data."""
        np.random.seed(seed)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = np.maximum(prices, 1.0)  # keep positive
        dates = pd.date_range("2024-01-01", periods=n, freq="B")

        highs = pd.Series(prices * 1.005, index=dates)
        lows = pd.Series(prices * 0.995, index=dates)
        closes = pd.Series(prices, index=dates)
        volumes = pd.Series(np.random.randint(1000000, 5000000, n).astype(float), index=dates)

        return highs, lows, closes, volumes

    def test_sma(self):
        from marketmaster.engines.technical import sma
        prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
        assert sma(prices, 3) == pytest.approx(4.0)  # (3+4+5)/3

    def test_sma_insufficient_data(self):
        from marketmaster.engines.technical import sma
        prices = pd.Series([1, 2], dtype=float)
        assert sma(prices, 5) is None

    def test_ema(self):
        from marketmaster.engines.technical import ema
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = ema(prices, 10)
        assert result is not None
        assert 5 < result < 10  # Should be between mean and last value

    def test_rsi_all_gains(self):
        from marketmaster.engines.technical import rsi
        prices = pd.Series(range(1, 25), dtype=float)  # monotonically increasing
        result = rsi(prices, 14)
        assert result is not None
        assert result > 90  # Should be near 100 for all gains

    def test_rsi_all_losses(self):
        from marketmaster.engines.technical import rsi
        prices = pd.Series(range(25, 1, -1), dtype=float)  # monotonically decreasing
        result = rsi(prices, 14)
        assert result is not None
        assert result < 10  # Should be near 0 for all losses

    def test_rsi_insufficient_data(self):
        from marketmaster.engines.technical import rsi
        prices = pd.Series([1, 2, 3], dtype=float)
        assert rsi(prices, 14) is None

    def test_macd(self):
        from marketmaster.engines.technical import macd
        highs, lows, closes, _ = self._make_price_data(250)
        result = macd(closes)
        assert result is not None
        macd_line, signal_line, histogram = result
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_macd_insufficient(self):
        from marketmaster.engines.technical import macd
        closes = pd.Series([1, 2, 3], dtype=float)
        assert macd(closes) is None

    def test_atr(self):
        from marketmaster.engines.technical import atr
        highs, lows, closes, _ = self._make_price_data(250)
        result = atr(highs, lows, closes, 14)
        assert result is not None
        assert result > 0

    def test_adx(self):
        from marketmaster.engines.technical import adx
        highs, lows, closes, _ = self._make_price_data(100)
        result = adx(highs, lows, closes, 14)
        assert result is not None
        assert 0 <= result <= 100

    def test_bollinger_bands(self):
        from marketmaster.engines.technical import bollinger_bands
        closes = pd.Series(range(1, 25), dtype=float)
        result = bollinger_bands(closes, 20, 2.0)
        assert result is not None
        upper, middle, lower, bandwidth = result
        assert upper > middle > lower
        assert bandwidth > 0

    def test_relative_strength(self):
        from marketmaster.engines.technical import relative_strength
        prices = pd.Series(np.linspace(100, 110, 60), dtype=float)
        benchmark = pd.Series(np.linspace(100, 102, 60), dtype=float)
        result = relative_strength(prices, benchmark, 60)
        assert result is not None
        assert result > 1  # security outperformed

    def test_compute_all_technical(self):
        from marketmaster.engines.technical import compute_all_technical
        highs, lows, closes, volumes = self._make_price_data(250)
        results = compute_all_technical(highs, lows, closes, volumes)
        assert "rsi_14" in results
        assert "sma_20" in results
        assert "adx_14" in results
        assert "macd" in results
        assert "bollinger_upper" in results
        assert "volume_ratio" in results
        assert len(results) >= 20

    def test_volume_ratio(self):
        from marketmaster.engines.technical import volume_ratio
        volumes = pd.Series([100] * 20 + [200], dtype=float)
        result = volume_ratio(volumes, 20)
        assert result == 2.0  # 200 / 100

    def test_momentum(self):
        from marketmaster.engines.technical import momentum
        prices = pd.Series([100, 105, 110], dtype=float)
        result = momentum(prices, 2)
        assert result == pytest.approx(0.10)  # 110/100 - 1 = 0.10


class TestMCEIPipeline:
    """Test MCEI pipeline computation logic."""

    def test_transform_pct_yoy(self):
        """Test that the YoY transform produces expected results."""
        from marketmaster.engines.mcei_pipeline import MceiPipeline

        # Simulate monthly data
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        values = pd.Series([100 + i for i in range(24)], index=dates, dtype=float)

        # We can test the transform method statically
        pipeline = MceiPipeline.__new__(MceiPipeline)  # avoid __init__ (needs db)

        result = MceiPipeline._apply_transform(pipeline, values, "pct_yoy")
        assert not result.empty
        # The 13th observation (index 12) should be ~12% YoY growth
        # (112 - 100) / 100 = 0.12
        assert abs(result.iloc[12] - 12.0) < 1.0

    def test_transform_level(self):
        from marketmaster.engines.mcei_pipeline import MceiPipeline
        pipeline = MceiPipeline.__new__(MceiPipeline)
        values = pd.Series([1, 2, 3], dtype=float)
        result = MceiPipeline._apply_transform(pipeline, values, "level")
        assert result.equals(values)

    def test_sign_alignment(self):
        from marketmaster.engines.mcei_pipeline import MceiPipeline
        pipeline = MceiPipeline.__new__(MceiPipeline)
        values = pd.Series([75.0, 25.0])

        # Positive sign: keep as-is
        pos = MceiPipeline._apply_sign(pipeline, values, "pos")
        assert pos.iloc[0] == 75.0

        # Negative sign: invert (100 - value)
        neg = MceiPipeline._apply_sign(pipeline, values, "neg")
        assert neg.iloc[0] == 25.0

    def test_normalize_component(self):
        from marketmaster.engines.mcei_pipeline import MceiPipeline
        pipeline = MceiPipeline.__new__(MceiPipeline)

        # A series where the latest value is the max
        values = pd.Series([10, 20, 30, 40, 50], dtype=float)
        result = MceiPipeline._normalize_component(pipeline, values)
        assert result.iloc[-1] > 90  # Should be near 100 (highest value)

        # A series where the latest value is the min
        # With 5 values, the minimum has a percentile rank of 1/5 = 20%
        values = pd.Series([50, 40, 30, 20, 10], dtype=float)
        result = MceiPipeline._normalize_component(pipeline, values)
        assert result.iloc[-1] <= 25  # Should be low (lowest value, ~20%)

    def test_classify_regime(self):
        from marketmaster.engines.mcei_pipeline import MceiPipeline
        pipeline = MceiPipeline.__new__(MceiPipeline)

        # 85 >= 80 (STRONG_EXPANSION) -> STRONG_BULL
        assert MceiPipeline._classify_regime(pipeline, 85) == "STRONG_BULL"
        # 70 >= 60 (EXPANSION) -> BULL
        assert MceiPipeline._classify_regime(pipeline, 70) == "BULL"
        # 50 >= 40 (NEUTRAL) -> NEUTRAL
        assert MceiPipeline._classify_regime(pipeline, 50) == "NEUTRAL"
        # 15 < 20 (below CONTRACTION) -> CRISIS
        assert MceiPipeline._classify_regime(pipeline, 15) == "CRISIS"


class TestThresholdOptimizer:
    """Test threshold optimizer logic."""

    def test_classify_with_thresholds(self):
        from marketmaster.engines.threshold_optimizer import ThresholdOptimizer
        from marketmaster.db.models import MceiHistory

        scores = pd.Series([85, 65, 50, 30, 10], dtype=float)
        opt = ThresholdOptimizer.__new__(ThresholdOptimizer)

        regimes = ThresholdOptimizer._classify_with_thresholds(
            opt, scores,
            [("STRONG_BULL", 80, 101), ("BULL", 60, 80), ("NEUTRAL", 45, 60),
             ("BEAR", 20, 45), ("CRISIS", 0, 20)],
        )

        assert regimes.iloc[0] == "STRONG_BULL"
        assert regimes.iloc[1] == "BULL"
        assert regimes.iloc[2] == "NEUTRAL"
        assert regimes.iloc[3] == "BEAR"
        assert regimes.iloc[4] == "CRISIS"

    def test_compute_separation_quality(self):
        from marketmaster.engines.threshold_optimizer import ThresholdOptimizer

        # Good separation: clear difference in returns by regime
        regimes = pd.Series(["BULL"] * 10 + ["BEAR"] * 10)
        returns = pd.Series([0.05, 0.06, 0.04, 0.07, 0.03, 0.05, 0.06, 0.04, 0.05, 0.03,
                            -0.03, -0.02, -0.04, -0.03, -0.02, -0.04, -0.03, -0.02, -0.04, -0.03])

        opt = ThresholdOptimizer.__new__(ThresholdOptimizer)
        quality, stats = ThresholdOptimizer._compute_separation_quality(opt, regimes, returns)

        assert quality > 0  # Should have positive separation
        assert stats["BULL"]["mean_return"] > 0
        assert stats["BEAR"]["mean_return"] < 0

    def test_compute_separation_no_separation(self):
        from marketmaster.engines.threshold_optimizer import ThresholdOptimizer

        # No separation: all returns similar
        regimes = pd.Series(["BULL", "BEAR"])
        returns = pd.Series([0.01, 0.01])

        opt = ThresholdOptimizer.__new__(ThresholdOptimizer)
        quality, _ = ThresholdOptimizer._compute_separation_quality(opt, regimes, returns)
        assert quality <= 0.01  # Near zero — no separation


class TestScoring:
    """Test the scoring engine."""

    def test_opportunity_score_equal_weights(self):
        from marketmaster.engines.scoring import opportunity_score
        scores = {"a": 0.8, "b": 0.6}
        weights = {"a": 0.5, "b": 0.5}
        result = opportunity_score(scores, weights)
        assert result == pytest.approx(0.7)

    def test_opportunity_score_default_weights(self):
        from marketmaster.engines.scoring import opportunity_score, DEFAULT_WEIGHTS
        # All categories scored equally
        scores = {k: 0.5 for k in DEFAULT_WEIGHTS}
        result = opportunity_score(scores)
        assert result == pytest.approx(0.5)

    def test_opportunity_score_missing_keys(self):
        from marketmaster.engines.scoring import opportunity_score
        scores = {"momentum": 1.0}  # Only one score
        result = opportunity_score(scores)
        assert 0 <= result <= 1


class TestRiskGate:
    """Test the deterministic risk gate."""

    def test_position_risk_limit(self):
        from marketmaster.risk.gate import risk_gate
        result = risk_gate(0.02, 0.005, 0.01, 0.02, True)
        assert not result.approved
        assert "POSITION_RISK_LIMIT" in result.reasons

    def test_daily_loss_limit(self):
        from marketmaster.risk.gate import risk_gate
        result = risk_gate(0.001, 0.005, 0.025, 0.02, True)
        assert not result.approved
        assert "DAILY_LOSS_LIMIT" in result.reasons

    def test_live_trading_disabled(self):
        from marketmaster.risk.gate import risk_gate
        result = risk_gate(0.001, 0.005, 0.001, 0.02, False)
        assert not result.approved
        assert "LIVE_TRADING_DISABLED" in result.reasons

    def test_all_clear(self):
        from marketmaster.risk.gate import risk_gate
        result = risk_gate(0.001, 0.005, 0.001, 0.02, True)
        assert result.approved
        assert len(result.reasons) == 0
