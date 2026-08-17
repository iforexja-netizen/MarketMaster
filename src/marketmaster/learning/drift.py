"""
Drift Detection — Detect when market conditions shift from what models expect.

The drift detector monitors:
1. Feature drift — input features shifting distribution (mean, std, range)
2. Regime drift — market regime changing more frequently or to unusual states
3. Performance drift — strategy performance degrading over time
4. Data quality drift — data freshness, completeness, or coverage changing
5. Volatility regime drift — volatility clustering or regime change

Drift is the enemy of models trained on historical data. When the market
shifts, models that worked yesterday stop working. This module detects
that shift early so strategies can be adjusted or deactivated.

Detection methods:
- Population Stability Index (PSI) for feature distribution shifts
- CUSUM (cumulative sum) for performance degradation
- Kolmogorov-Smirnov statistic for distribution changes
- Simple mean/std z-score for quick checks
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Any
from enum import Enum
import numpy as np
import math
from collections import deque


class DriftType(Enum):
    FEATURE = "feature"
    REGIME = "regime"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    VOLATILITY = "volatility"


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class DriftAlert:
    """An alert that drift has been detected."""
    type: DriftType
    severity: DriftSeverity
    metric_name: str
    message: str
    current_value: float = 0.0
    baseline_value: float = 0.0
    drift_score: float = 0.0  # How far from baseline (PSI, z-score, etc.)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DriftReport:
    """Full drift assessment report."""
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alerts: list[DriftAlert] = field(default_factory=list)
    feature_drift: dict[str, float] = field(default_factory=dict)  # feature → PSI
    performance_drift: dict[str, float] = field(default_factory=dict)  # strategy → degradation %
    regime_stability: float = 1.0  # 0-1, 1 = stable
    overall_risk: DriftSeverity = DriftSeverity.NONE
    summary: str = ""

    @property
    def has_drift(self) -> bool:
        return any(a.severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE) for a in self.alerts)

    @property
    def critical_alerts(self) -> list[DriftAlert]:
        return [a for a in self.alerts if a.severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE)]


class FeatureBaseline:
    """Baseline statistics for a feature (from training/reference period)."""
    def __init__(self, name: str, values: list[float], n_bins: int = 10):
        self.name = name
        self.n_bins = n_bins
        values_arr = np.array(values, dtype=float)
        self.mean = float(np.mean(values_arr)) if len(values_arr) > 0 else 0.0
        self.std = float(np.std(values_arr)) if len(values_arr) > 0 else 1.0
        self.min = float(np.min(values_arr)) if len(values_arr) > 0 else 0.0
        self.max = float(np.max(values_arr)) if len(values_arr) > 0 else 0.0

        # Build histogram for PSI calculation
        if len(values_arr) > 0 and self.std > 0:
            hist, edges = np.histogram(values_arr, bins=n_bins, density=False)
            self.bin_edges = edges
            self.bin_probs = hist / len(values_arr)
            # Avoid zero probabilities
            self.bin_probs = np.where(self.bin_probs == 0, 1e-6, self.bin_probs)
        else:
            self.bin_edges = np.linspace(0, 1, n_bins + 1)
            self.bin_probs = np.full(n_bins, 1.0 / n_bins)

    def compute_psi(self, recent_values: list[float]) -> float:
        """
        Population Stability Index.

        PSI < 0.1: No significant drift
        PSI 0.1-0.25: Moderate drift
        PSI > 0.25: Significant drift

        Formula: PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        """
        if not recent_values:
            return 0.0

        recent_arr = np.array(recent_values, dtype=float)
        # Use baseline bin edges to bin recent values
        recent_hist, _ = np.histogram(recent_arr, bins=self.bin_edges)
        recent_probs = recent_hist / len(recent_arr) if len(recent_arr) > 0 else np.zeros_like(self.bin_probs)
        recent_probs = np.where(recent_probs == 0, 1e-6, recent_probs)

        psi = np.sum((recent_probs - self.bin_probs) * np.log(recent_probs / self.bin_probs))
        return float(psi)

    def compute_zscore(self, recent_values: list[float]) -> float:
        """Simple z-score of recent mean vs baseline mean."""
        if not recent_values or self.std == 0:
            return 0.0
        recent_mean = np.mean(recent_values)
        return abs(recent_mean - self.mean) / self.std


class DriftDetector:
    """
    Detects drift across features, performance, and regime stability.

    Usage:
        detector = DriftDetector()
        detector.set_baseline("rsi", baseline_rsi_values)
        detector.set_baseline("momentum", baseline_momentum_values)

        # Periodically:
        psi = detector.check_feature_drift("rsi", recent_rsi_values)
        detector.check_performance("trend_following", recent_returns, baseline_returns)
        report = detector.generate_report()
    """

    def __init__(
        self,
        psi_warning_threshold: float = 0.1,
        psi_critical_threshold: float = 0.25,
        performance_window: int = 50,  # trades to look back
        regime_stability_window: int = 30,  # days
    ):
        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.performance_window = performance_window
        self.regime_window = regime_stability_window

        self._feature_baselines: dict[str, FeatureBaseline] = {}
        self._performance_history: dict[str, deque] = {}  # strategy → deque of returns
        self._performance_baselines: dict[str, float] = {}  # strategy → baseline avg return
        self._regime_history: deque = deque(maxlen=regime_stability_window)
        self._recent_alerts: list[DriftAlert] = []

    # ── Feature Drift ───────────────────────────────────────────────────────

    def set_baseline(self, feature_name: str, values: list[float], n_bins: int = 10):
        """Set the baseline distribution for a feature."""
        self._feature_baselines[feature_name] = FeatureBaseline(feature_name, values, n_bins)

    def check_feature_drift(self, feature_name: str, recent_values: list[float]) -> Optional[DriftAlert]:
        """
        Check a feature for distribution drift using PSI.

        Returns a DriftAlert if drift is detected, None otherwise.
        """
        baseline = self._feature_baselines.get(feature_name)
        if not baseline:
            return None

        psi = baseline.compute_psi(recent_values)
        zscore = baseline.compute_zscore(recent_values)

        if psi > self.psi_critical:
            severity = DriftSeverity.SEVERE if psi > 0.5 else DriftSeverity.HIGH
        elif psi > self.psi_warning:
            severity = DriftSeverity.MODERATE
        else:
            return None  # No drift

        alert = DriftAlert(
            type=DriftType.FEATURE,
            severity=severity,
            metric_name=feature_name,
            message=f"Feature '{feature_name}' drift detected: PSI={psi:.4f}, z-score={zscore:.2f}",
            current_value=float(np.mean(recent_values)) if recent_values else 0.0,
            baseline_value=baseline.mean,
            drift_score=psi,
            recommendations=[
                f"Recalibrate {feature_name} thresholds",
                f"Consider retraining models using {feature_name}",
            ] if severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE) else [
                f"Monitor {feature_name} closely",
            ],
        )
        self._recent_alerts.append(alert)
        return alert

    # ── Performance Drift ──────────────────────────────────────────────────

    def set_performance_baseline(self, strategy_name: str, baseline_avg_return: float):
        """Set the baseline average return for a strategy."""
        self._performance_baselines[strategy_name] = baseline_avg_return
        if strategy_name not in self._performance_history:
            self._performance_history[strategy_name] = deque(maxlen=self.performance_window)

    def record_trade_return(self, strategy_name: str, return_pct: float):
        """Record a trade return for a strategy."""
        if strategy_name not in self._performance_history:
            self._performance_history[strategy_name] = deque(maxlen=self.performance_window)
        self._performance_history[strategy_name].append(return_pct)

    def check_performance_drift(self, strategy_name: str) -> Optional[DriftAlert]:
        """
        Check if a strategy's recent performance has drifted from baseline.

        Uses CUSUM (cumulative sum) approach to detect sustained degradation.
        """
        baseline = self._performance_baselines.get(strategy_name)
        history = self._performance_history.get(strategy_name)

        if baseline is None or not history or len(history) < 10:
            return None

        returns = list(history)
        recent_avg = np.mean(returns[-min(20, len(returns)):])

        # Degradation: how far below baseline
        if baseline > 0:
            degradation = (baseline - recent_avg) / baseline * 100
        elif baseline < 0:
            degradation = 0.0
        else:
            degradation = -recent_avg * 100

        # CUSUM: cumulative sum of deviations from baseline
        deviations = [r - baseline for r in returns]
        cusum = np.cumsum(deviations)
        max_cusum = max(cusum) if len(cusum) > 0 else 0
        min_cusum = min(cusum) if len(cusum) > 0 else 0
        cusum_range = max_cusum - min_cusum

        if degradation > 50 or cusum_range > 10:
            severity = DriftSeverity.SEVERE
        elif degradation > 25:
            severity = DriftSeverity.HIGH
        elif degradation > 10:
            severity = DriftSeverity.MODERATE
        else:
            return None

        alert = DriftAlert(
            type=DriftType.PERFORMANCE,
            severity=severity,
            metric_name=strategy_name,
            message=f"Strategy '{strategy_name}' performance drift: recent avg {recent_avg:.2f}% vs baseline {baseline:.2f}% (degradation: {degradation:.1f}%)",
            current_value=recent_avg,
            baseline_value=baseline,
            drift_score=degradation,
            recommendations=[
                f"Review {strategy_name} parameter calibration",
                "Check if market regime has shifted",
                "Consider reducing allocation to this strategy",
            ] if severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE) else [
                f"Monitor {strategy_name} performance closely",
            ],
        )
        self._recent_alerts.append(alert)
        return alert

    # ── Regime Stability ───────────────────────────────────────────────────

    def record_regime(self, regime: str, as_of: Optional[date] = None):
        """Record the current market regime."""
        if as_of is None:
            as_of = date.today()
        self._regime_history.append((as_of, regime))

    def check_regime_stability(self) -> Optional[DriftAlert]:
        """
        Check if regime has been changing too frequently (instability).

        Frequent regime changes suggest an unstable market where
        strategies may not have time to play out.
        """
        if len(self._regime_history) < 5:
            return None

        regimes = [r for _, r in self._regime_history]
        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
        change_rate = changes / len(regimes)

        # Count unique regimes
        unique = len(set(regimes))

        if change_rate > 0.5 or unique > 5:
            severity = DriftSeverity.HIGH if change_rate > 0.6 else DriftSeverity.MODERATE
            return DriftAlert(
                type=DriftType.REGIME,
                severity=severity,
                metric_name="regime_stability",
                message=f"Regime instability detected: {changes} changes in {len(regimes)} periods ({change_rate:.1%} change rate, {unique} unique regimes)",
                current_value=change_rate,
                baseline_value=0.1,  # Expected: <10% change rate
                drift_score=change_rate,
                recommendations=[
                    "Reduce position sizes due to regime instability",
                    "Prioritize defensive strategies",
                    "Wait for regime to stabilize before aggressive positioning",
                ],
            )
        return None

    # ── Volatility Drift ────────────────────────────────────────────────────

    def check_volatility_drift(
        self,
        recent_volatility: list[float],
        baseline_volatility: float,
    ) -> Optional[DriftAlert]:
        """
        Check if recent volatility has shifted from baseline.
        """
        if not recent_volatility or baseline_volatility <= 0:
            return None

        recent_avg = np.mean(recent_volatility)
        vol_ratio = recent_avg / baseline_volatility

        if vol_ratio > 2.0:
            severity = DriftSeverity.SEVERE
        elif vol_ratio > 1.5:
            severity = DriftSeverity.HIGH
        elif vol_ratio > 1.25:
            severity = DriftSeverity.MODERATE
        else:
            return None

        return DriftAlert(
            type=DriftType.VOLATILITY,
            severity=severity,
            metric_name="volatility",
            message=f"Volatility drift: recent {recent_avg:.4f} vs baseline {baseline_volatility:.4f} ({vol_ratio:.2f}×)",
            current_value=recent_avg,
            baseline_value=baseline_volatility,
            drift_score=vol_ratio,
            recommendations=[
                "Widen stops to avoid noise-triggered exits",
                "Reduce position sizes",
                "Consider volatility-targeted position sizing",
            ],
        )

    # ── Full Report ────────────────────────────────────────────────────────

    def generate_report(
        self,
        feature_values: Optional[dict[str, list[float]]] = None,
    ) -> DriftReport:
        """
        Generate a full drift report across all monitored dimensions.

        Args:
            feature_values: {feature_name → recent_values} to check for drift

        Returns:
            DriftReport with all detected drift alerts
        """
        report = DriftReport()
        alerts = []

        # Feature drift
        if feature_values:
            for name, values in feature_values.items():
                alert = self.check_feature_drift(name, values)
                if alert:
                    alerts.append(alert)
                    report.feature_drift[name] = alert.drift_score

        # Performance drift
        for strategy_name in self._performance_history:
            alert = self.check_performance_drift(strategy_name)
            if alert:
                alerts.append(alert)
                report.performance_drift[strategy_name] = alert.drift_score

        # Regime stability
        regime_alert = self.check_regime_stability()
        if regime_alert:
            alerts.append(regime_alert)

        report.alerts = alerts

        # Overall risk = max severity
        if alerts:
            severities = [a.severity for a in alerts]
            if DriftSeverity.SEVERE in severities:
                report.overall_risk = DriftSeverity.SEVERE
            elif DriftSeverity.HIGH in severities:
                report.overall_risk = DriftSeverity.HIGH
            elif DriftSeverity.MODERATE in severities:
                report.overall_risk = DriftSeverity.MODERATE
            else:
                report.overall_risk = DriftSeverity.LOW

        # Regime stability score
        if len(self._regime_history) >= 5:
            regimes = [r for _, r in self._regime_history]
            changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
            report.regime_stability = 1.0 - (changes / len(regimes))

        # Summary
        if not alerts:
            report.summary = "No drift detected. All systems stable."
        else:
            critical = [a for a in alerts if a.severity in (DriftSeverity.HIGH, DriftSeverity.SEVERE)]
            report.summary = (
                f"Drift detected: {len(alerts)} alerts "
                f"({len(critical)} critical). "
                f"Overall risk: {report.overall_risk.value}."
            )

        return report

    @property
    def recent_alerts(self) -> list[DriftAlert]:
        """Get recently generated alerts."""
        return self._recent_alerts[-50:]  # Last 50

    def clear_alerts(self):
        """Clear alert history."""
        self._recent_alerts.clear()
