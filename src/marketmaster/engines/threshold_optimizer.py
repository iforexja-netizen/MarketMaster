"""
Walk-Forward Threshold Optimizer

Tests MCEI regime thresholds against historical data to validate or
discover the boundaries that best separate market regimes.

Key principle: "80 = bull" is a hypothesis, not a truth — until the data
confirms it through walk-forward testing.

Methodology:
1. Split MCEI history into rolling train/test windows
2. For each train window, optimize threshold boundaries that maximize
   regime separation quality (measured by forward returns per regime)
3. Test those thresholds on the out-of-sample test window
4. Measure stability: how much do optimal thresholds vary across windows?
5. Report the robustness of each threshold boundary

This prevents overfitting — we never optimize on the same data we test on.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from marketmaster.db.models import MceiHistory


# Default thresholds (hypothesis, not truth)
DEFAULT_THRESHOLDS = [
    ("STRONG_BULL", 80, 101),
    ("BULL", 60, 80),
    ("TRANSITION_BULL", 55, 60),
    ("NEUTRAL", 45, 55),
    ("TRANSITION_BEAR", 40, 45),
    ("BEAR", 20, 40),
    ("CRISIS", 0, 20),
]


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class ThresholdResult:
    """Result of testing a set of thresholds on a window."""
    thresholds: list[tuple[str, float, float]]
    train_sharpe: float
    test_sharpe: float
    regime_returns: dict[str, float]
    regime_counts: dict[str, int]
    separation_quality: float  # higher = better regime separation


@dataclass
class OptimizerReport:
    """Full walk-forward optimization report."""
    windows: int
    optimal_thresholds: list[tuple[str, float, float]]
    threshold_stability: dict[str, float]  # std dev of each boundary across windows
    mean_train_sharpe: float
    mean_test_sharpe: float
    overfit_ratio: float  # train/test ratio — high = overfitting
    regime_summary: dict[str, dict[str, float]]  # regime -> {mean_return, std_return, sharpe, count}
    recommendation: str


class ThresholdOptimizer:
    """
    Walk-forward threshold optimizer for MCEI regime boundaries.

    Tests whether the default thresholds (or alternative boundaries)
    produce regimes that meaningfully separate forward market returns.

    A good regime classification should show:
    - STRONG_BULL: high positive forward returns
    - BEAR/CRISIS: negative forward returns
    - Clear difference between adjacent regimes
    """

    def __init__(
        self,
        db: Session,
        benchmark_symbol: str = "SPY",
        forward_return_days: list[int] = None,
    ):
        self.db = db
        self.benchmark_symbol = benchmark_symbol
        self.forward_return_days = forward_return_days or [21, 63, 126]  # 1M, 3M, 6M

    def _load_mcei_history(self) -> pd.DataFrame:
        """Load all MCEI history into a DataFrame."""
        stmt = select(MceiHistory).order_by(MceiHistory.as_of_date)
        records = list(self.db.execute(stmt).scalars().all())

        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            data.append({
                "date": r.as_of_date,
                "score": float(r.score),
                "regime": r.regime,
            })

        return pd.DataFrame(data).set_index("date").sort_index()

    def _load_forward_returns(
        self,
        mcei_dates: list[date],
        forward_days: int,
    ) -> pd.Series:
        """
        Compute forward returns for the benchmark on each MCEI date.

        This requires OHLCV data for the benchmark.
        """
        from marketmaster.data.plane import DataPlane
        from marketmaster.db.models import SecurityMaster, OhlcvDaily

        plane = DataPlane(self.db)
        sec = plane.get_security_by_symbol(self.benchmark_symbol)
        if not sec:
            return pd.Series(dtype=float)

        # Get all benchmark prices
        stmt = (
            select(OhlcvDaily)
            .where(OhlcvDaily.security_id == sec.id)
            .order_by(OhlcvDaily.date)
        )
        bars = list(self.db.execute(stmt).scalars().all())

        if len(bars) < 2:
            return pd.Series(dtype=float)

        prices = pd.Series(
            [float(b.close) if b.close else np.nan for b in bars],
            index=pd.DatetimeIndex([b.date for b in bars]),
        )

        # For each MCEI date, compute the forward return
        forward_returns = {}
        for mcei_date in mcei_dates:
            # Find the benchmark price on or after mcei_date
            ts = pd.Timestamp(mcei_date)
            mask_before = prices.index <= ts
            if not mask_before.any():
                continue

            entry_price = prices.loc[mask_before].iloc[-1]
            if np.isnan(entry_price):
                continue

            # Find the price forward_days later
            entry_idx = prices.index.get_loc(prices.loc[mask_before].index[-1])
            exit_idx = entry_idx + forward_days

            if exit_idx >= len(prices):
                continue

            exit_price = prices.iloc[exit_idx]
            if np.isnan(exit_price) or entry_price == 0:
                continue

            forward_returns[mcei_date] = (exit_price / entry_price) - 1.0

        return pd.Series(forward_returns)

    def _classify_with_thresholds(
        self,
        scores: pd.Series,
        thresholds: list[tuple[str, float, float]],
    ) -> pd.Series:
        """Classify scores into regimes using given thresholds."""
        regimes = pd.Series(index=scores.index, dtype=str)

        for regime_name, low, high in thresholds:
            mask = (scores >= low) & (scores < high)
            regimes.loc[mask] = regime_name

        # Fill any gaps with NEUTRAL
        regimes = regimes.fillna("NEUTRAL")

        return regimes

    def _compute_separation_quality(
        self,
        regimes: pd.Series,
        forward_returns: pd.Series,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        """
        Compute how well the regimes separate forward returns.

        A good classification has:
        - Monotonic ordering of mean returns across regimes
        - Low within-regime variance relative to between-regime variance
        """
        aligned = pd.DataFrame({"regime": regimes, "return": forward_returns}).dropna()

        if len(aligned) < 10:
            return 0.0, {}

        regime_stats = {}
        regime_means = []

        for regime in sorted(aligned["regime"].unique()):
            mask = aligned["regime"] == regime
            returns = aligned.loc[mask, "return"]
            if len(returns) < 3:
                continue
            regime_stats[regime] = {
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()) if len(returns) > 1 else 0.0,
                "sharpe": float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
                "count": int(len(returns)),
            }
            regime_means.append(returns.mean())

        if len(regime_means) < 2:
            return 0.0, regime_stats

        # Separation quality: ratio of between-regime variance to total variance
        between_var = np.var(regime_means)
        total_var = aligned["return"].var()
        if total_var == 0:
            return 0.0, regime_stats

        quality = between_var / total_var

        # Bonus for monotonic ordering (bull regimes have higher returns)
        # Check if the ordering is roughly monotonic
        expected_order = ["CRISIS", "BEAR", "TRANSITION_BEAR", "NEUTRAL",
                          "TRANSITION_BULL", "BULL", "STRONG_BULL"]
        ordered_means = [regime_stats.get(r, {}).get("mean_return", 0) for r in expected_order if r in regime_stats]
        if len(ordered_means) >= 2:
            # Count direction changes
            diffs = np.diff(ordered_means)
            monotonic_score = sum(1 for d in diffs if d > 0) / len(diffs)
            quality = quality * (0.5 + 0.5 * monotonic_score)

        return float(quality), regime_stats

    def _optimize_thresholds(
        self,
        scores: pd.Series,
        forward_returns: pd.Series,
    ) -> tuple[list[tuple[str, float, float]], float]:
        """
        Find threshold boundaries that maximize regime separation quality.

        Uses a grid search over the key boundaries (bear/neutral and neutral/bull).
        """
        best_quality = -1
        best_thresholds = DEFAULT_THRESHOLDS

        # Grid search over the critical boundaries
        bear_neutral_range = np.arange(30, 50, 2.5)  # bear/neutral boundary
        neutral_bull_range = np.arange(50, 70, 2.5)  # neutral/bull boundary
        bull_strong_range = np.arange(70, 90, 2.5)  # bull/strong_bull boundary

        for bn in bear_neutral_range:
            for nb in neutral_bull_range:
                for bs in bull_strong_range:
                    thresholds = [
                        ("STRONG_BULL", float(bs), 101),
                        ("BULL", float(nb), float(bs)),
                        ("TRANSITION_BULL", float(nb - 5), float(nb)),
                        ("NEUTRAL", float(bn), float(nb - 5)),
                        ("TRANSITION_BEAR", float(bn - 5), float(bn)),
                        ("BEAR", float(bn - 20), float(bn - 5)),
                        ("CRISIS", 0, float(bn - 20)),
                    ]

                    regimes = self._classify_with_thresholds(scores, thresholds)
                    quality, _ = self._compute_separation_quality(regimes, forward_returns)

                    if quality > best_quality:
                        best_quality = quality
                        best_thresholds = thresholds

        return best_thresholds, float(best_quality)

    def run_walk_forward(
        self,
        train_years: int = 10,
        test_years: int = 3,
        step_years: int = 1,
        forward_days: int = 63,  # 3-month forward returns
    ) -> OptimizerReport:
        """
        Run full walk-forward threshold optimization.

        Args:
            train_years: Training window size in years
            test_years: Test window size in years
            step_years: How far to step forward between windows
            forward_days: Days of forward returns to measure

        Returns an OptimizerReport with stability analysis.
        """
        mcei_df = self._load_mcei_history()

        if len(mcei_df) < 100:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation="Insufficient MCEI history for walk-forward analysis. "
                             f"Need at least 100 observations, have {len(mcei_df)}. "
                             "Run FRED backfill and MCEI computation first.",
            )

        # Compute forward returns for all MCEI dates
        forward_returns = self._load_forward_returns(
            list(mcei_df.index), forward_days
        )

        if len(forward_returns) < 50:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation="Insufficient forward return data. "
                             "Need benchmark OHLCV data covering the MCEI history period.",
            )

        # Align the data
        aligned = mcei_df.join(pd.DataFrame({"forward_return": forward_returns}))
        aligned = aligned.dropna(subset=["score", "forward_return"])

        if len(aligned) < 50:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation="Insufficient overlapping MCEI and forward return data.",
            )

        # Generate walk-forward windows
        min_date = aligned.index.min()
        max_date = aligned.index.max()
        total_days = (max_date - min_date).days

        if total_days < (train_years + test_years) * 365:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation=f"Insufficient date range. Need at least "
                             f"{train_years + test_years} years of data, have "
                             f"{total_days / 365:.1f} years.",
            )

        windows: list[WalkForwardWindow] = []
        train_delta = timedelta(days=train_years * 365)
        test_delta = timedelta(days=test_years * 365)
        step_delta = timedelta(days=step_years * 365)

        current = min_date
        while current + train_delta + test_delta <= max_date:
            windows.append(WalkForwardWindow(
                train_start=current.date(),
                train_end=(current + train_delta).date(),
                test_start=(current + train_delta).date(),
                test_end=(current + train_delta + test_delta).date(),
            ))
            current += step_delta

        if not windows:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation="Could not generate any walk-forward windows.",
            )

        # Run optimization on each window
        results: list[ThresholdResult] = []
        boundary_values: dict[str, list[float]] = {
            "bear_neutral": [],
            "neutral_bull": [],
            "bull_strong": [],
        }

        for w in windows:
            train_data = aligned.loc[
                (aligned.index >= pd.Timestamp(w.train_start)) &
                (aligned.index < pd.Timestamp(w.train_end))
            ]
            test_data = aligned.loc[
                (aligned.index >= pd.Timestamp(w.test_start)) &
                (aligned.index < pd.Timestamp(w.test_end))
            ]

            if len(train_data) < 30 or len(test_data) < 10:
                continue

            # Optimize on train
            opt_thresholds, train_quality = self._optimize_thresholds(
                train_data["score"], train_data["forward_return"]
            )

            # Test on test window
            train_regimes = self._classify_with_thresholds(train_data["score"], opt_thresholds)
            test_regimes = self._classify_with_thresholds(test_data["score"], opt_thresholds)
            train_quality_full, train_stats = self._compute_separation_quality(train_regimes, train_data["forward_return"])
            test_quality, test_stats = self._compute_separation_quality(test_regimes, test_data["forward_return"])

            result = ThresholdResult(
                thresholds=opt_thresholds,
                train_sharpe=train_quality_full,
                test_sharpe=test_quality,
                regime_returns={k: v.get("mean_return", 0) for k, v in train_stats.items()},
                regime_counts={k: int(v.get("count", 0)) for k, v in train_stats.items()},
                separation_quality=test_quality,
            )
            results.append(result)

            # Track boundary stability
            for regime_name, low, high in opt_thresholds:
                if regime_name == "BEAR":
                    boundary_values["bear_neutral"].append(float(high))
                elif regime_name == "NEUTRAL":
                    boundary_values["neutral_bull"].append(float(high))
                elif regime_name == "BULL":
                    boundary_values["bull_strong"].append(float(high))

        if not results:
            return OptimizerReport(
                windows=0,
                optimal_thresholds=DEFAULT_THRESHOLDS,
                threshold_stability={},
                mean_train_sharpe=0.0,
                mean_test_sharpe=0.0,
                overfit_ratio=0.0,
                regime_summary={},
                recommendation="No windows had sufficient data for optimization.",
            )

        # Compute stability
        threshold_stability = {}
        for name, values in boundary_values.items():
            if values:
                threshold_stability[name] = float(np.std(values))

        mean_train = float(np.mean([r.train_sharpe for r in results]))
        mean_test = float(np.mean([r.test_sharpe for r in results]))
        overfit = mean_train / mean_test if mean_test > 0 else float("inf")

        # Aggregate regime summary across all test windows
        regime_summary: dict[str, dict[str, float]] = {}
        all_regimes_set = set()
        for r in results:
            all_regimes_set.update(r.regime_returns.keys())

        for regime in sorted(all_regimes_set):
            returns = [r.regime_returns.get(regime, 0) for r in results if regime in r.regime_returns]
            counts = [r.regime_counts.get(regime, 0) for r in results if regime in r.regime_counts]
            if returns:
                regime_summary[regime] = {
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)) if len(returns) > 1 else 0.0,
                    "total_count": int(np.sum(counts)),
                    "window_count": len(returns),
                }

        # Generate recommendation
        max_stability = max(threshold_stability.values()) if threshold_stability else 0
        if overfit > 3.0:
            recommendation = (
                "OVERFITTING WARNING: Train/test ratio is high "
                f"({overfit:.1f}x). The optimized thresholds are likely fitting noise. "
                "Consider using wider regime bands or fewer regime classifications."
            )
        elif max_stability > 10:
            recommendation = (
                "INSTABILITY WARNING: Threshold boundaries vary significantly "
                f"across walk-forward windows (max std={max_stability:.1f}). "
                "The default thresholds may be as good as optimized ones."
            )
        elif mean_test > 0.3:
            recommendation = (
                "GOOD SEPARATION: Regime classification shows meaningful forward return "
                f"separation (test quality={mean_test:.2f}). Thresholds are reasonably stable. "
                "Safe to use for strategy selection."
            )
        else:
            recommendation = (
                "WEAK SEPARATION: Regimes show limited forward return differentiation "
                f"(test quality={mean_test:.2f}). Consider adjusting the MCEI components "
                "or using fewer, broader regime classifications."
            )

        return OptimizerReport(
            windows=len(results),
            optimal_thresholds=results[-1].thresholds if results else DEFAULT_THRESHOLDS,
            threshold_stability=threshold_stability,
            mean_train_sharpe=mean_train,
            mean_test_sharpe=mean_test,
            overfit_ratio=overfit,
            regime_summary=regime_summary,
            recommendation=recommendation,
        )
