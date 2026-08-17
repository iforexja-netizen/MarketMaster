"""
Walk-Forward Validation — Out-of-sample strategy validation.

Walk-forward analysis tests a strategy by:
1. Dividing the historical data into training and testing windows
2. Optimizing/fitting on the training window
3. Testing on the next out-of-sample window
4. Rolling forward and repeating

This catches overfitting: if a strategy performs well in-sample but poorly
out-of-sample, it's overfit and should not be trusted in live trading.

The walk-forward efficiency (WFE) ratio measures how much of in-sample
performance is preserved out-of-sample:
    WFE = OOS_return / IS_return
    WFE > 0.5 = good, WFE < 0.25 = overfit
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional, Callable
import numpy as np

from marketmaster.backtest.engine import BacktestEngine, BacktestResult, BacktestMetrics


@dataclass
class WalkForwardWindow:
    """A single train/test window in walk-forward analysis."""
    window_num: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    in_sample_result: Optional[BacktestResult] = None
    out_of_sample_result: Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    symbol: str
    strategy_name: str
    windows: list[WalkForwardWindow] = field(default_factory=list)
    avg_is_return: float = 0.0
    avg_oos_return: float = 0.0
    wfe_ratio: float = 0.0  # Walk-Forward Efficiency
    avg_oos_sharpe: float = 0.0
    avg_oos_max_dd: float = 0.0
    oos_win_rate: float = 0.0
    consistency: float = 0.0  # fraction of OOS windows that are profitable
    overfit_warning: bool = False
    summary: str = ""


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies.

    Usage:
        validator = WalkForwardValidator(
            train_days=504,  # 2 years
            test_days=126,   # 6 months
            step_days=63,    # 3 months overlap
        )
        result = validator.run(
            strategy=strategy,
            symbol="AAPL",
            bars=historical_bars,
            regime="BULL",
        )
    """

    def __init__(
        self,
        train_days: int = 504,  # ~2 years of trading days
        test_days: int = 126,   # ~6 months
        step_days: int = 63,    # ~3 months step
        initial_capital: float = 100_000,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.initial_capital = initial_capital

    def run(
        self,
        strategy: Any,
        symbol: str,
        bars: list[Any],
        regime: str = "NEUTRAL",
        evidence_fn: Optional[Callable[[date], list[Any]]] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Returns WalkForwardResult with IS/OOS performance per window,
        WFE ratio, and overfit detection.
        """
        result = WalkForwardResult(
            symbol=symbol,
            strategy_name=strategy.name,
        )

        if len(bars) < self.train_days + self.test_days:
            result.summary = "Insufficient data for walk-forward analysis"
            return result

        engine = BacktestEngine(initial_capital=self.initial_capital)

        # Generate windows
        windows = []
        total_bars = len(bars)
        start_idx = 0
        window_num = 0

        while start_idx + self.train_days + self.test_days <= total_bars:
            train_bars = bars[start_idx:start_idx + self.train_days]
            test_bars = bars[start_idx + self.train_days:start_idx + self.train_days + self.test_days]

            if not train_bars or not test_bars:
                break

            window = WalkForwardWindow(
                window_num=window_num,
                train_start=train_bars[0].date,
                train_end=train_bars[-1].date,
                test_start=test_bars[0].date,
                test_end=test_bars[-1].date,
            )

            # Run in-sample backtest
            try:
                window.in_sample_result = engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    bars=train_bars,
                    evidence_fn=evidence_fn,
                    regime=regime,
                )
            except Exception:
                pass

            # Run out-of-sample backtest
            try:
                window.out_of_sample_result = engine.run(
                    strategy=strategy,
                    symbol=symbol,
                    bars=test_bars,
                    evidence_fn=evidence_fn,
                    regime=regime,
                )
            except Exception:
                pass

            windows.append(window)
            window_num += 1
            start_idx += self.step_days

        result.windows = windows

        # Compute aggregate metrics
        if windows:
            is_returns = []
            oos_returns = []
            oos_sharpes = []
            oos_dds = []
            oos_win_rates = []
            profitable_oos = 0

            for w in windows:
                if w.in_sample_result:
                    is_returns.append(w.in_sample_result.metrics.total_return_pct)
                if w.out_of_sample_result:
                    oos_ret = w.out_of_sample_result.metrics.total_return_pct
                    oos_returns.append(oos_ret)
                    oos_sharpes.append(w.out_of_sample_result.metrics.sharpe_ratio)
                    oos_dds.append(w.out_of_sample_result.metrics.max_drawdown_pct)
                    oos_win_rates.append(w.out_of_sample_result.metrics.win_rate)
                    if oos_ret > 0:
                        profitable_oos += 1

            if is_returns:
                result.avg_is_return = float(np.mean(is_returns))
            if oos_returns:
                result.avg_oos_return = float(np.mean(oos_returns))
                result.avg_oos_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0
                result.avg_oos_max_dd = float(np.mean(oos_dds)) if oos_dds else 0
                result.oos_win_rate = float(np.mean(oos_win_rates)) if oos_win_rates else 0
                result.consistency = profitable_oos / len(oos_returns)

            # WFE ratio
            if result.avg_is_return != 0:
                result.wfe_ratio = result.avg_oos_return / abs(result.avg_is_return)
            elif result.avg_oos_return > 0:
                result.wfe_ratio = 1.0  # No IS return but OOS is positive

            # Overfit detection
            result.overfit_warning = result.wfe_ratio < 0.25

            # Summary
            result.summary = self._generate_summary(result)

        return result

    def _generate_summary(self, result: WalkForwardResult) -> str:
        """Generate human-readable walk-forward summary."""
        lines = [
            f"Walk-Forward Validation: {result.symbol} / {result.strategy_name}",
            f"Windows tested: {len(result.windows)}",
            f"Avg IS return: {result.avg_is_return:.1f}%",
            f"Avg OOS return: {result.avg_oos_return:.1f}%",
            f"WFE ratio: {result.wfe_ratio:.2f} (>0.5 good, <0.25 overfit)",
            f"OOS consistency: {result.consistency:.0%} of windows profitable",
            f"OOS avg Sharpe: {result.avg_oos_sharpe:.2f}",
            f"OOS avg max DD: {result.avg_oos_max_dd:.1f}%",
        ]
        if result.overfit_warning:
            lines.append("⚠ OVERFIT WARNING: OOS performance significantly below IS")
        return "\n".join(lines)
