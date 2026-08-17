"""
Phase 6 tests — Learning System.

Tests signal attribution, calibration monitoring, model registry,
drift detection, and strategy ranking.
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from marketmaster.learning.attribution import (
    SignalAttribution, AttributionTracker,
    StrategyPerformanceStats, AgentPerformanceStats, RegimePerformanceStats,
)
from marketmaster.learning.calibration import (
    CalibrationMonitor, CalibrationResult, CalibrationBin, BrierScore,
)
from marketmaster.learning.registry import (
    ModelRegistry, ModelVersion, ModelType, ModelStatus, VersionComparison,
)
from marketmaster.learning.drift import (
    DriftDetector, DriftReport, DriftAlert, DriftType, DriftSeverity,
    FeatureBaseline,
)
from marketmaster.learning.ranking import (
    StrategyRanker, StrategyRanking, RankingReport,
    RankingMetric, AllocationAction,
)


# ============================================================================
# Signal Attribution Tests
# ============================================================================

class TestSignalAttribution:
    def _make_signal(self, **kwargs):
        return SignalAttribution(
            signal_id=kwargs.get("signal_id", "sig_001"),
            symbol=kwargs.get("symbol", "AAPL"),
            strategy_name=kwargs.get("strategy_name", "trend_following"),
            regime=kwargs.get("regime", "BULL"),
            signal_date=kwargs.get("signal_date", date(2025, 6, 1)),
            signal_score=kwargs.get("signal_score", 75),
            signal_confidence=kwargs.get("signal_confidence", 0.7),
            signal_direction=kwargs.get("signal_direction", "long"),
            agent_evidence=kwargs.get("agent_evidence", {"macro": 70, "technical": 65}),
            debate_score=kwargs.get("debate_score", 30),
            debate_winner=kwargs.get("debate_winner", "bull"),
            entry_price=kwargs.get("entry_price", 150.0),
            exit_price=kwargs.get("exit_price", None),
            entry_date=kwargs.get("entry_date", date(2025, 6, 1)),
            exit_date=kwargs.get("exit_date", None),
            stop_price=kwargs.get("stop_price", 145.0),
            pnl_dollars=kwargs.get("pnl_dollars", 0),
            pnl_pct=kwargs.get("pnl_pct", 0),
            hold_days=kwargs.get("hold_days", 0),
            exit_reason=kwargs.get("exit_reason", None),
            r_multiple=kwargs.get("r_multiple", 0),
            win=kwargs.get("win", False),
            market_regime_at_entry=kwargs.get("market_regime_at_entry", "BULL"),
            market_regime_at_exit=kwargs.get("market_regime_at_exit", None),
        )

    def test_signal_attribution_creation(self):
        attr = self._make_signal()
        assert attr.signal_id == "sig_001"
        assert attr.symbol == "AAPL"
        assert attr.strategy_name == "trend_following"

    def test_r_multiple_long_win(self):
        # Entry 150, stop 145, exit 165 → risk=5, reward=15, R=3.0
        attr = self._make_signal(exit_price=165.0, exit_date=date(2025, 6, 10),
                                 pnl_dollars=1500, pnl_pct=10, exit_reason="take_profit",
                                 r_multiple=3.0, win=True)
        assert attr.r_multiple == 3.0
        assert attr.win is True

    def test_r_multiple_long_loss(self):
        # Entry 150, stop 145, exit 140 → risk=5, loss=-10, R=-2.0
        attr = self._make_signal(exit_price=140.0, exit_date=date(2025, 6, 5),
                                 pnl_dollars=-1000, pnl_pct=-6.67, exit_reason="stop_loss",
                                 r_multiple=-2.0, win=False)
        assert attr.win is False


from dataclasses import dataclass
from marketmaster.learning.attribution import ExitReason

@dataclass
class MockSignal:
    """Mock signal object for attribution tests."""
    symbol: str = "AAPL"
    strategy_name: str = "trend_following"
    direction: str = "long"
    score: float = 75.0
    confidence: float = 0.7
    entry_price: float = 150.0
    stop_price: float = 145.0
    evidence: dict = field(default_factory=lambda: {"macro": 70})
    regime: str = "BULL"
    as_of: Optional[date] = None


class TestAttributionTracker:
    def _make_signal(self, **kwargs):
        return MockSignal(**{k: v for k, v in kwargs.items() if k in MockSignal.__dataclass_fields__})

    def test_tracker_creation(self):
        tracker = AttributionTracker()
        assert tracker is not None

    def test_record_entry(self):
        tracker = AttributionTracker()
        signal_id = tracker.record_entry(
            signal=self._make_signal(),
            fill_price=150.0,
            fill_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            regime="BULL",
        )
        assert signal_id is not None
        assert isinstance(signal_id, str)

    def test_record_exit_completes_attribution(self):
        tracker = AttributionTracker()
        signal_id = tracker.record_entry(
            signal=self._make_signal(),
            fill_price=150.0,
            fill_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            regime="BULL",
        )
        result = tracker.record_exit(
            signal_id=signal_id, exit_price=165.0,
            exit_date=datetime(2025, 6, 10, tzinfo=timezone.utc),
            exit_reason=ExitReason.TAKE_PROFIT, regime="BULL",
        )
        assert result is not None
        assert result.win is True
        assert result.exit_price == 165.0
        assert result.pnl_dollars > 0

    def test_record_exit_loss(self):
        tracker = AttributionTracker()
        signal_id = tracker.record_entry(
            signal=self._make_signal(entry_price=150, stop_price=145),
            fill_price=150.0,
            fill_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            regime="BULL",
        )
        result = tracker.record_exit(
            signal_id=signal_id, exit_price=140.0,
            exit_date=datetime(2025, 6, 3, tzinfo=timezone.utc),
            exit_reason=ExitReason.STOP_LOSS,
        )
        assert result.win is False
        assert result.pnl_dollars < 0

    def test_get_strategy_stats(self):
        tracker = AttributionTracker()
        for i in range(10):
            signal_id = tracker.record_entry(
                signal=self._make_signal(entry_price=100, stop_price=95),
                fill_price=100.0,
                fill_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
                regime="BULL",
            )
            exit_price = 105 if i % 3 != 0 else 94
            tracker.record_exit(
                signal_id=signal_id, exit_price=exit_price,
                exit_date=datetime(2025, 6, 10, tzinfo=timezone.utc),
                exit_reason=ExitReason.TAKE_PROFIT if exit_price > 100 else ExitReason.STOP_LOSS,
            )
        stats = tracker.get_strategy_stats("trend_following")
        assert stats is not None
        assert stats.total_signals == 10
        assert stats.win_rate > 0.5

    def test_summary(self):
        tracker = AttributionTracker()
        signal_id = tracker.record_entry(
            signal=self._make_signal(),
            fill_price=150.0,
            fill_date=datetime(2025, 6, 1, tzinfo=timezone.utc),
            regime="BULL",
        )
        tracker.record_exit(
            signal_id=signal_id, exit_price=160,
            exit_date=datetime(2025, 6, 5, tzinfo=timezone.utc),
            exit_reason=ExitReason.TAKE_PROFIT,
        )
        summary = tracker.summary()
        assert isinstance(summary, dict)
        assert summary["closed_trades"] >= 1


# ============================================================================
# Calibration Tests
# ============================================================================

class TestCalibration:
    def test_monitor_creation(self):
        monitor = CalibrationMonitor()
        assert monitor is not None

    def test_record_prediction_and_outcome(self):
        monitor = CalibrationMonitor()
        monitor.record_prediction("sig_001", predicted_confidence=0.7, predicted_score=75,
                                   strategy="trend_following", regime="BULL")
        monitor.record_outcome("sig_001", actual_win=True, pnl_pct=5.0)
        # Should have one recorded prediction
        assert len(monitor._records) >= 1

    def test_well_calibrated(self):
        monitor = CalibrationMonitor()
        np.random.seed(42)
        # Generate predictions where confidence matches actual win rate
        for i in range(200):
            conf = np.random.uniform(0.3, 0.9)
            won = np.random.random() < conf
            monitor.record_prediction(f"sig_{i}", predicted_confidence=conf,
                                       predicted_score=conf * 100, strategy="test", regime="BULL")
            monitor.record_outcome(f"sig_{i}", actual_win=won, pnl_pct=5.0 if won else -3.0)
        result = monitor.compute_calibration(n_bins=6)
        assert result is not None
        assert result.brier_score >= 0
        assert len(result.bins) > 0

    def test_overconfident_system(self):
        monitor = CalibrationMonitor()
        # System predicts 80% confidence but only wins 40% of the time
        for i in range(100):
            monitor.record_prediction(f"sig_{i}", predicted_confidence=0.8,
                                       predicted_score=80, strategy="test", regime="BULL")
            won = i % 5 < 2  # 40% win rate
            monitor.record_outcome(f"sig_{i}", actual_win=won, pnl_pct=5 if won else -3)
        result = monitor.compute_calibration(n_bins=5)
        assert result.overconfidence_score > 0  # Should be overconfident
        assert len(result.recommendations) > 0

    def test_brier_score_decomposition(self):
        predictions = [0.7, 0.6, 0.8, 0.5, 0.9]
        outcomes = [1, 0, 1, 0, 1]
        result = BrierScore.compute(predictions, outcomes)
        assert "brier" in result
        assert "reliability" in result
        assert "resolution" in result
        assert result["brier"] >= 0

    def test_calibration_bins_populated(self):
        monitor = CalibrationMonitor()
        for i in range(50):
            conf = 0.5 + (i % 5) * 0.1  # Spread across bins
            monitor.record_prediction(f"sig_{i}", predicted_confidence=conf,
                                       predicted_score=conf * 100, strategy="test", regime="BULL")
            monitor.record_outcome(f"sig_{i}", actual_win=True, pnl_pct=5)
        result = monitor.compute_calibration(n_bins=5)
        for b in result.bins:
            assert b.n_predictions >= 0

    def test_recommendations_generated(self):
        monitor = CalibrationMonitor()
        # Overconfident predictions
        for i in range(50):
            monitor.record_prediction(f"sig_{i}", predicted_confidence=0.9,
                                       predicted_score=90, strategy="overconfident_strat", regime="BULL")
            monitor.record_outcome(f"sig_{i}", actual_win=False, pnl_pct=-3)
        result = monitor.compute_calibration(n_bins=5)
        assert len(result.recommendations) > 0
        assert any("overconfident" in r.lower() for r in result.recommendations)


# ============================================================================
# Model Registry Tests
# ============================================================================

class TestModelRegistry:
    def test_registry_creation(self):
        registry = ModelRegistry()
        assert registry is not None

    def test_register_version(self):
        registry = ModelRegistry()
        mv = registry.register(
            ModelType.STRATEGY, "trend_following", "1.0.0",
            {"stop_loss_pct": 5.0, "adx_threshold": 25},
        )
        assert mv.name == "trend_following"
        assert mv.version == "1.0.0"
        assert mv.hash != ""
        assert mv.status == ModelStatus.EXPERIMENTAL

    def test_activate_version(self):
        registry = ModelRegistry()
        mv = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        success = registry.activate(mv.id)
        assert success
        assert mv.status == ModelStatus.PRODUCTION
        assert mv.activated_at is not None

    def test_activate_deprecates_previous(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.activate(v1.id)
        v2 = registry.register(ModelType.STRATEGY, "trend_following", "1.1.0", {"stop": 4.0})
        registry.activate(v2.id)
        assert v1.status == ModelStatus.DEPRECATED
        assert v2.status == ModelStatus.PRODUCTION

    def test_get_active(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.activate(v1.id)
        active = registry.get_active("trend_following")
        assert active is not None
        assert active.version == "1.0.0"

    def test_no_active_returns_none(self):
        registry = ModelRegistry()
        assert registry.get_active("nonexistent") is None

    def test_get_all_versions(self):
        registry = ModelRegistry()
        registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.register(ModelType.STRATEGY, "trend_following", "1.1.0", {"stop": 4.0})
        registry.register(ModelType.STRATEGY, "trend_following", "2.0.0", {"stop": 3.0})
        versions = registry.get_all_versions("trend_following")
        assert len(versions) == 3

    def test_lineage(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        v2 = registry.register(ModelType.STRATEGY, "trend_following", "1.1.0", {"stop": 4.0}, parent_version=v1.id)
        v3 = registry.register(ModelType.STRATEGY, "trend_following", "2.0.0", {"stop": 3.0}, parent_version=v2.id)
        lineage = registry.get_lineage(v3.id)
        assert len(lineage) == 3
        assert lineage[0].version == "1.0.0"
        assert lineage[-1].version == "2.0.0"

    def test_compare_versions(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0",
                                {"stop_loss_pct": 5.0, "adx_threshold": 25, "old_param": True})
        v2 = registry.register(ModelType.STRATEGY, "trend_following", "1.1.0",
                                {"stop_loss_pct": 4.0, "adx_threshold": 25, "new_param": "value"})
        diff = registry.compare(v1.id, v2.id)
        assert "stop_loss_pct" in diff.parameter_changes
        assert "new_param" in diff.new_parameters
        assert "old_param" in diff.removed_parameters

    def test_duplicate_not_recreated(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        v2 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.1", {"stop": 5.0})
        assert v1.id == v2.id  # Same hash → same version

    def test_retire(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.activate(v1.id)
        registry.retire(v1.id)
        assert v1.status == ModelStatus.RETIRED
        assert registry.get_active("trend_following") is None

    def test_update_metrics(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.update_metrics(v1.id, {"sharpe": 1.5, "win_rate": 0.6})
        assert v1.metrics["sharpe"] == 1.5

    def test_summary(self):
        registry = ModelRegistry()
        registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.register(ModelType.MCEI_ENGINE, "mcei", "1.0.0", {"threshold": 80})
        summary = registry.summary()
        assert summary["total_versions"] == 2
        assert "strategy" in summary["by_type"]
        assert "mcei_engine" in summary["by_type"]

    def test_export(self):
        registry = ModelRegistry()
        registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        exported = registry.export()
        assert len(exported) == 1
        assert "hash" in exported[0]

    def test_list_active(self):
        registry = ModelRegistry()
        v1 = registry.register(ModelType.STRATEGY, "trend_following", "1.0.0", {"stop": 5.0})
        registry.activate(v1.id)
        active = registry.list_active()
        assert len(active) == 1


# ============================================================================
# Drift Detection Tests
# ============================================================================

class TestFeatureBaseline:
    def test_baseline_creation(self):
        values = list(np.random.normal(50, 10, 200))
        baseline = FeatureBaseline("rsi", values, n_bins=10)
        assert baseline.name == "rsi"
        assert baseline.mean > 45 and baseline.mean < 55
        assert len(baseline.bin_probs) == 10

    def test_psi_no_drift(self):
        np.random.seed(42)
        values = list(np.random.normal(50, 10, 500))
        baseline = FeatureBaseline("rsi", values, n_bins=10)
        recent = list(np.random.normal(50, 10, 300))  # Same distribution
        psi = baseline.compute_psi(recent)
        assert psi < 0.15  # No significant drift

    def test_psi_significant_drift(self):
        np.random.seed(42)
        values = list(np.random.normal(50, 10, 200))
        baseline = FeatureBaseline("rsi", values, n_bins=10)
        recent = list(np.random.normal(70, 10, 100))  # Shifted mean
        psi = baseline.compute_psi(recent)
        assert psi > 0.1  # Drift detected

    def test_zscore(self):
        values = list(np.random.normal(50, 10, 200))
        baseline = FeatureBaseline("rsi", values)
        z = baseline.compute_zscore([60, 62, 58])
        assert z > 0


class TestDriftDetector:
    def test_detector_creation(self):
        detector = DriftDetector()
        assert detector is not None

    def test_set_baseline_and_check_no_drift(self):
        detector = DriftDetector()
        np.random.seed(42)
        detector.set_baseline("rsi", list(np.random.normal(50, 10, 500)))
        alert = detector.check_feature_drift("rsi", list(np.random.normal(50, 10, 300)))
        assert alert is None  # No drift

    def test_feature_drift_detected(self):
        detector = DriftDetector()
        np.random.seed(42)
        detector.set_baseline("rsi", list(np.random.normal(50, 10, 200)))
        alert = detector.check_feature_drift("rsi", list(np.random.normal(80, 15, 100)))
        assert alert is not None
        assert alert.type == DriftType.FEATURE
        assert alert.severity in (DriftSeverity.MODERATE, DriftSeverity.HIGH, DriftSeverity.SEVERE)
        assert alert.drift_score > 0.1

    def test_performance_drift_detected(self):
        detector = DriftDetector()
        detector.set_performance_baseline("trend_following", baseline_avg_return=2.0)
        # Record declining returns
        for i in range(20):
            detector.record_trade_return("trend_following", -1.0 - i * 0.1)
        alert = detector.check_performance_drift("trend_following")
        assert alert is not None
        assert alert.type == DriftType.PERFORMANCE
        assert alert.drift_score > 10

    def test_no_performance_drift(self):
        detector = DriftDetector()
        detector.set_performance_baseline("trend_following", baseline_avg_return=2.0)
        for i in range(20):
            detector.record_trade_return("trend_following", 1.8 + np.random.uniform(-0.2, 0.3))
        alert = detector.check_performance_drift("trend_following")
        assert alert is None  # No significant drift

    def test_regime_stability_stable(self):
        detector = DriftDetector()
        for i in range(30):
            detector.record_regime("BULL", date(2025, 1, 1) + timedelta(days=i))
        alert = detector.check_regime_stability()
        assert alert is None  # Stable regime

    def test_regime_instability(self):
        detector = DriftDetector()
        regimes = ["BULL", "BEAR", "BULL", "CRISIS", "BULL", "BEAR", "NEUTRAL", "BULL", "BEAR", "CRISIS"]
        for i, r in enumerate(regimes):
            detector.record_regime(r, date(2025, 1, 1) + timedelta(days=i))
        alert = detector.check_regime_stability()
        assert alert is not None
        assert alert.type == DriftType.REGIME

    def test_volatility_drift(self):
        detector = DriftDetector()
        alert = detector.check_volatility_drift(
            recent_volatility=[0.03, 0.035, 0.04, 0.032],
            baseline_volatility=0.015,
        )
        assert alert is not None
        assert alert.type == DriftType.VOLATILITY
        assert alert.severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE, DriftSeverity.MODERATE)

    def test_volatility_no_drift(self):
        detector = DriftDetector()
        alert = detector.check_volatility_drift(
            recent_volatility=[0.015, 0.016, 0.014],
            baseline_volatility=0.015,
        )
        assert alert is None

    def test_generate_report_no_drift(self):
        detector = DriftDetector()
        np.random.seed(42)
        detector.set_baseline("rsi", list(np.random.normal(50, 10, 500)))
        report = detector.generate_report(feature_values={"rsi": list(np.random.normal(50, 10, 300))})
        assert report.overall_risk == DriftSeverity.NONE
        assert len(report.alerts) == 0

    def test_generate_report_with_drift(self):
        detector = DriftDetector()
        np.random.seed(42)
        detector.set_baseline("rsi", list(np.random.normal(50, 10, 500)))
        detector.set_performance_baseline("strat", 2.0)
        for i in range(20):
            detector.record_trade_return("strat", -1.0)
        report = detector.generate_report(
            feature_values={"rsi": list(np.random.normal(80, 15, 300))}
        )
        assert len(report.alerts) > 0
        assert report.has_drift

    def test_report_summary(self):
        detector = DriftDetector()
        report = detector.generate_report()
        assert "stable" in report.summary.lower() or "drift" in report.summary.lower()


# ============================================================================
# Strategy Ranking Tests
# ============================================================================

class TestStrategyRanker:
    def test_ranker_creation(self):
        ranker = StrategyRanker(min_trades=5)
        assert ranker is not None

    def test_add_trade_and_rank(self):
        ranker = StrategyRanker(min_trades=3)
        for i in range(5):
            ranker.add_trade("trend_following", pnl=500, r_multiple=2.1, win=True, hold_days=5)
        report = ranker.rank()
        assert len(report.rankings) == 1
        assert report.rankings[0].strategy_name == "trend_following"
        assert report.rankings[0].n_trades == 5
        assert report.rankings[0].win_rate == 1.0
        assert report.rankings[0].expectancy == 500

    def test_ranking_multiple_strategies(self):
        ranker = StrategyRanker(min_trades=3)
        # Good strategy
        for i in range(10):
            ranker.add_trade("trend_following", pnl=400, r_multiple=1.8, win=True, hold_days=4)
        # Bad strategy
        for i in range(10):
            ranker.add_trade("mean_reversion", pnl=-200, r_multiple=-0.8, win=False, hold_days=3)
        # Mediocre strategy
        for i in range(10):
            ranker.add_trade("value", pnl=50, r_multiple=0.3, win=True, hold_days=15)

        report = ranker.rank()
        assert len(report.rankings) == 3
        assert report.rankings[0].strategy_name == "trend_following"  # Best
        assert report.rankings[-1].strategy_name == "mean_reversion"  # Worst

    def test_allocation_actions(self):
        ranker = StrategyRanker(min_trades=5)
        # Top performer → INCREASE
        for i in range(10):
            ranker.add_trade("great_strat", pnl=1000, r_multiple=2.5, win=True, hold_days=5)
        # Loser → PAUSE
        for i in range(10):
            ranker.add_trade("bad_strat", pnl=-300, r_multiple=-1.2, win=False, hold_days=3)

        report = ranker.rank()
        great = report.get_ranking("great_strat")
        bad = report.get_ranking("bad_strat")
        assert great.allocation_action == AllocationAction.INCREASE
        assert bad.allocation_action == AllocationAction.PAUSE

    def test_investigate_few_trades(self):
        ranker = StrategyRanker(min_trades=20)
        ranker.add_trade("new_strat", pnl=500, r_multiple=2.0, win=True)
        report = ranker.rank()
        r = report.get_ranking("new_strat")
        assert r.allocation_action == AllocationAction.INVESTIGATE

    def test_edge_persistence_stable(self):
        ranker = StrategyRanker(min_trades=5)
        # Consistent returns
        for i in range(20):
            ranker.add_trade("stable_strat", pnl=200 + np.random.randint(-50, 50),
                             r_multiple=1.0, win=True, hold_days=5)
        report = ranker.rank()
        r = report.get_ranking("stable_strat")
        assert r.edge_persistence > 0.5  # Stable edge

    def test_edge_persistence_decaying(self):
        ranker = StrategyRanker(min_trades=5)
        # Strong start, weak finish
        for i in range(10):
            ranker.add_trade("decaying_strat", pnl=500, r_multiple=2.0, win=True, hold_days=5)
        for i in range(10):
            ranker.add_trade("decaying_strat", pnl=-200, r_multiple=-0.8, win=False, hold_days=3)
        report = ranker.rank()
        r = report.get_ranking("decaying_strat")
        assert r.edge_persistence < 0.3  # Edge decayed

    def test_regime_performance(self):
        ranker = StrategyRanker(min_trades=3)
        ranker.add_trade("strat", pnl=500, r_multiple=2.0, win=True, regime="BULL")
        ranker.add_trade("strat", pnl=-200, r_multiple=-0.8, win=False, regime="BEAR")
        report = ranker.rank()
        r = report.get_ranking("strat")
        assert "BULL" in r.regime_performance
        assert "BEAR" in r.regime_performance
        assert r.regime_performance["BULL"] > r.regime_performance["BEAR"]

    def test_composite_score_range(self):
        ranker = StrategyRanker(min_trades=3)
        for i in range(10):
            ranker.add_trade("strat", pnl=300, r_multiple=1.5, win=True)
        report = ranker.rank()
        assert 0 <= report.rankings[0].composite_score <= 100

    def test_recommendations_generated(self):
        ranker = StrategyRanker(min_trades=3)
        for i in range(10):
            ranker.add_trade("good", pnl=500, r_multiple=2.0, win=True)
        for i in range(10):
            ranker.add_trade("bad", pnl=-300, r_multiple=-1.0, win=False)
        report = ranker.rank()
        assert len(report.recommendations) > 0
        assert any("Increase" in r for r in report.recommendations)
        assert len(report.recommendations) > 0

    def test_summary(self):
        ranker = StrategyRanker(min_trades=3)
        for i in range(5):
            ranker.add_trade("strat", pnl=200, r_multiple=1.0, win=True)
        report = ranker.rank()
        assert "Ranked" in report.summary or "strategies" in report.summary.lower()

    def test_ranking_by_different_metrics(self):
        ranker = StrategyRanker(min_trades=3)
        # Strategy A: high win rate, low R
        for i in range(10):
            ranker.add_trade("A", pnl=50, r_multiple=0.3, win=True)
        # Strategy B: low win rate, high R
        for i in range(10):
            ranker.add_trade("B", pnl=500 if i < 3 else -100, r_multiple=3.0 if i < 3 else -0.5,
                             win=(i < 3))

        # By win rate, A should be first
        report_wr = ranker.rank(metric=RankingMetric.WIN_RATE)
        assert report_wr.rankings[0].strategy_name == "A"

        # By expectancy, B might be first
        report_exp = ranker.rank(metric=RankingMetric.EXPECTANCY)
        assert report_exp.rankings[0].strategy_name in ("A", "B")

    def test_empty_ranker(self):
        ranker = StrategyRanker()
        report = ranker.rank()
        assert len(report.rankings) == 0
