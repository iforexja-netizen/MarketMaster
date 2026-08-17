"""
Strategy Ranking — Rank strategies by realized performance, not backtests.

The ranking system uses the Signal Attribution Framework to compute
realized performance metrics for each strategy, then ranks them by:
1. Expectancy (average $ per trade)
2. R-multiple (average risk multiples per trade)
3. Win rate
4. Profit factor (gross profit / gross loss)
5. Sharpe estimate (return / volatility of returns)
6. Consistency (how stable are returns across trades)
7. Regime-specific performance (does it work in the current regime?)
8. Edge persistence (is the edge stable or decaying?)

Strategies that underperform consistently get ranked down.
Strategies that outperform get ranked up.
The system recommends allocation adjustments based on ranking.

Key principle: past performance doesn't guarantee future results,
but consistent underperformance is a signal to investigate.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Any
from enum import Enum
import numpy as np
from collections import defaultdict


class RankingMetric(Enum):
    EXPECTANCY = "expectancy"
    R_MULTIPLE = "r_multiple"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE = "sharpe"
    CONSISTENCY = "consistency"
    COMPOSITE = "composite"


class AllocationAction(Enum):
    INCREASE = "increase"
    MAINTAIN = "maintain"
    DECREASE = "decrease"
    PAUSE = "pause"
    INVESTIGATE = "investigate"


@dataclass
class StrategyRanking:
    """Ranking of a single strategy by realized performance."""
    rank: int
    strategy_name: str
    n_trades: int
    expectancy: float = 0.0       # avg $ per trade
    avg_r_multiple: float = 0.0  # avg risk multiples per trade
    win_rate: float = 0.0        # 0-1
    profit_factor: float = 0.0   # gross profit / gross loss
    sharpe_estimate: float = 0.0 # return / volatility
    consistency: float = 0.0      # 0-1, how stable returns are
    total_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_hold_days: float = 0.0
    composite_score: float = 0.0  # weighted combination of all metrics
    edge_persistence: float = 0.0  # 0-1, is the edge stable or decaying?
    allocation_action: AllocationAction = AllocationAction.MAINTAIN
    allocation_reason: str = ""
    regime_performance: dict[str, float] = field(default_factory=dict)  # regime → avg R


@dataclass
class RankingReport:
    """Full strategy ranking report."""
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rankings: list[StrategyRanking] = field(default_factory=list)
    metric_used: RankingMetric = RankingMetric.COMPOSITE
    total_strategies: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    best_strategy: str = ""
    worst_strategy: str = ""
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""

    def get_ranking(self, strategy_name: str) -> Optional[StrategyRanking]:
        for r in self.rankings:
            if r.strategy_name == strategy_name:
                return r
        return None


class StrategyRanker:
    """
    Ranks strategies by realized performance from attribution data.

    Usage:
        ranker = StrategyRanker()
        ranker.add_trade("trend_following", pnl=500, r_multiple=2.1, win=True, regime="BULL")
        ranker.add_trade("mean_reversion", pnl=-100, r_multiple=-0.5, win=False, regime="BULL")
        report = ranker.rank()
    """

    def __init__(
        self,
        min_trades: int = 10,
        composite_weights: Optional[dict] = None,
        edge_decay_window: int = 30,  # trades to look back for edge persistence
    ):
        self.min_trades = min_trades
        self.edge_decay_window = edge_decay_window

        # Default composite weights
        self.weights = composite_weights or {
            "expectancy": 0.25,
            "r_multiple": 0.20,
            "win_rate": 0.15,
            "profit_factor": 0.15,
            "sharpe": 0.15,
            "consistency": 0.10,
        }

        # Trade history per strategy
        self._trades: dict[str, list[dict]] = defaultdict(list)

    def add_trade(
        self,
        strategy_name: str,
        pnl: float,
        r_multiple: float,
        win: bool,
        hold_days: int = 0,
        pnl_pct: float = 0.0,
        regime: str = "unknown",
        trade_date: Optional[date] = None,
    ):
        """Record a completed trade for a strategy."""
        self._trades[strategy_name].append({
            "pnl": pnl,
            "r_multiple": r_multiple,
            "win": win,
            "hold_days": hold_days,
            "pnl_pct": pnl_pct,
            "regime": regime,
            "date": trade_date or date.today(),
        })

    def add_trades_from_attributions(self, attributions: list) -> int:
        """
        Add trades from SignalAttribution objects (from the attribution framework).

        Returns the number of trades added.
        """
        count = 0
        for attr in attributions:
            if attr.exit_price is not None and attr.entry_price is not None:
                self.add_trade(
                    strategy_name=attr.strategy_name,
                    pnl=attr.pnl_dollars,
                    r_multiple=attr.r_multiple,
                    win=attr.win,
                    hold_days=attr.hold_days,
                    pnl_pct=attr.pnl_pct,
                    regime=attr.market_regime_at_entry,
                )
                count += 1
        return count

    def rank(self, metric: RankingMetric = RankingMetric.COMPOSITE) -> RankingReport:
        """
        Rank all strategies by the specified metric.

        Strategies with fewer than min_trades are included but flagged
        with AllocationAction.INVESTIGATE (not enough data).
        """
        report = RankingReport(metric_used=metric, total_strategies=len(self._trades))

        rankings = []
        for strategy_name, trades in self._trades.items():
            ranking = self._compute_ranking(strategy_name, trades, metric)
            rankings.append(ranking)

        # Sort by the relevant metric (descending = best first)
        if metric == RankingMetric.COMPOSITE:
            rankings.sort(key=lambda r: r.composite_score, reverse=True)
        elif metric == RankingMetric.EXPECTANCY:
            rankings.sort(key=lambda r: r.expectancy, reverse=True)
        elif metric == RankingMetric.R_MULTIPLE:
            rankings.sort(key=lambda r: r.avg_r_multiple, reverse=True)
        elif metric == RankingMetric.WIN_RATE:
            rankings.sort(key=lambda r: r.win_rate, reverse=True)
        elif metric == RankingMetric.PROFIT_FACTOR:
            rankings.sort(key=lambda r: r.profit_factor, reverse=True)
        elif metric == RankingMetric.SHARPE:
            rankings.sort(key=lambda r: r.sharpe_estimate, reverse=True)
        elif metric == RankingMetric.CONSISTENCY:
            rankings.sort(key=lambda r: r.consistency, reverse=True)

        # Assign ranks
        for i, r in enumerate(rankings, 1):
            r.rank = i

        # Determine allocation actions
        for r in rankings:
            r.allocation_action, r.allocation_reason = self._determine_action(r)

        report.rankings = rankings
        report.total_trades = sum(r.n_trades for r in rankings)
        report.total_pnl = sum(r.total_pnl for r in rankings)

        if rankings:
            report.best_strategy = rankings[0].strategy_name
            report.worst_strategy = rankings[-1].strategy_name

        # Recommendations
        report.recommendations = self._generate_recommendations(rankings)
        report.summary = self._generate_summary(rankings)

        return report

    def _compute_ranking(
        self, strategy_name: str, trades: list[dict], metric: RankingMetric
    ) -> StrategyRanking:
        """Compute all metrics for a single strategy."""
        n = len(trades)
        pnls = [t["pnl"] for t in trades]
        r_multiples = [t["r_multiple"] for t in trades]
        wins = [t["win"] for t in trades]
        hold_days = [t["hold_days"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]

        # Basic metrics
        expectancy = float(np.mean(pnls)) if pnls else 0.0
        avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0
        win_rate = float(np.mean(wins)) if wins else 0.0
        total_pnl = float(np.sum(pnls))
        best_trade = float(max(pnls)) if pnls else 0.0
        worst_trade = float(min(pnls)) if pnls else 0.0
        avg_hold = float(np.mean(hold_days)) if hold_days else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Sharpe estimate (annualized, assuming ~252 trading days)
        if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
            sharpe = float(np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Consistency: how stable are returns?
        if len(pnls) > 1:
            # Coefficient of variation (lower = more consistent)
            std = np.std(pnls)
            mean = np.mean(pnls)
            if abs(mean) > 0:
                cv = abs(std / mean)
                consistency = max(0.0, min(1.0, 1.0 - cv))
            else:
                consistency = 0.0
        else:
            consistency = 0.0

        # Edge persistence: compare first half vs second half performance
        edge_persistence = self._compute_edge_persistence(pnls)

        # Regime performance
        regime_perf = defaultdict(list)
        for t in trades:
            regime_perf[t["regime"]].append(t["r_multiple"])
        regime_performance = {
            regime: float(np.mean(rs)) for regime, rs in regime_perf.items()
        }

        # Composite score (0-100)
        composite = self._compute_composite(
            expectancy, avg_r, win_rate, profit_factor, sharpe, consistency
        )

        return StrategyRanking(
            rank=0,  # Assigned later
            strategy_name=strategy_name,
            n_trades=n,
            expectancy=expectancy,
            avg_r_multiple=avg_r,
            win_rate=win_rate,
            profit_factor=min(profit_factor, 99.0),  # Cap for display
            sharpe_estimate=sharpe,
            consistency=consistency,
            total_pnl=total_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_hold_days=avg_hold,
            composite_score=composite,
            edge_persistence=edge_persistence,
            regime_performance=regime_performance,
        )

    def _compute_edge_persistence(self, pnls: list[float]) -> float:
        """
        Measure if the edge is stable or decaying.

        Compares first-half performance to second-half.
        Returns 0-1 where 1 = perfectly stable, 0 = completely decayed.
        """
        n = len(pnls)
        if n < 6:
            return 0.5  # Not enough data

        midpoint = n // 2
        first_half = np.mean(pnls[:midpoint])
        second_half = np.mean(pnls[midpoint:])

        if first_half == 0:
            return 0.5

        # Ratio of second half to first half
        if first_half > 0 and second_half > 0:
            ratio = second_half / first_half
            return min(1.0, max(0.0, ratio))
        elif first_half > 0 and second_half <= 0:
            return 0.0  # Edge completely reversed
        elif first_half < 0 and second_half > 0:
            return 0.7  # Improved from negative to positive
        else:
            return 0.5  # Both negative, unclear

    def _compute_composite(
        self, expectancy: float, avg_r: float, win_rate: float,
        profit_factor: float, sharpe: float, consistency: float
    ) -> float:
        """Compute weighted composite score (0-100)."""
        # Normalize each metric to 0-100
        # Expectancy: map -1000 to +1000 → 0 to 100
        exp_score = max(0, min(100, 50 + expectancy / 20))

        # R-multiple: map -3 to +3 → 0 to 100
        r_score = max(0, min(100, 50 + avg_r * 16.67))

        # Win rate: 0-1 → 0-100
        wr_score = win_rate * 100

        # Profit factor: 0-3 → 0-100 (cap at 3)
        pf_score = min(100, profit_factor / 3 * 100) if profit_factor < float('inf') else 100

        # Sharpe: -2 to +2 → 0 to 100
        sh_score = max(0, min(100, 50 + sharpe * 25))

        # Consistency: 0-1 → 0-100
        con_score = consistency * 100

        # Weighted average
        composite = (
            exp_score * self.weights["expectancy"] +
            r_score * self.weights["r_multiple"] +
            wr_score * self.weights["win_rate"] +
            pf_score * self.weights["profit_factor"] +
            sh_score * self.weights["sharpe"] +
            con_score * self.weights["consistency"]
        )

        return composite

    def _determine_action(self, ranking: StrategyRanking) -> tuple[AllocationAction, str]:
        """Determine what allocation action to recommend."""
        # Not enough trades
        if ranking.n_trades < self.min_trades:
            return AllocationAction.INVESTIGATE, (
                f"Only {ranking.n_trades} trades — need {self.min_trades}+ for reliable assessment"
            )

        # Edge decaying
        if ranking.edge_persistence < 0.3 and ranking.n_trades >= 20:
            return AllocationAction.INVESTIGATE, (
                "Edge persistence is low — strategy may be decaying"
            )

        # Consistently losing
        if ranking.expectancy < 0 and ranking.win_rate < 0.35:
            return AllocationAction.PAUSE, (
                f"Negative expectancy ({ranking.expectancy:.2f}) with low win rate ({ranking.win_rate:.0%})"
            )

        # Poor performance
        if ranking.composite_score < 30:
            return AllocationAction.DECREASE, (
                f"Composite score {ranking.composite_score:.0f} — underperforming"
            )

        # Underperforming but not terrible
        if ranking.composite_score < 45:
            return AllocationAction.DECREASE, (
                f"Composite score {ranking.composite_score:.0f} — below average"
            )

        # Average
        if ranking.composite_score < 60:
            return AllocationAction.MAINTAIN, (
                f"Composite score {ranking.composite_score:.0f} — adequate"
            )

        # Strong performer
        if ranking.composite_score < 75:
            return AllocationAction.MAINTAIN, (
                f"Composite score {ranking.composite_score:.0f} — performing well"
            )

        # Top performer
        return AllocationAction.INCREASE, (
            f"Composite score {ranking.composite_score:.0f} — strong performer"
        )

    def _generate_recommendations(self, rankings: list[StrategyRanking]) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        # Top performers
        top = [r for r in rankings if r.allocation_action == AllocationAction.INCREASE]
        if top:
            names = [r.strategy_name for r in top]
            recs.append(f"Increase allocation to: {', '.join(names)}")

        # Underperformers
        decrease = [r for r in rankings if r.allocation_action == AllocationAction.DECREASE]
        if decrease:
            names = [r.strategy_name for r in decrease]
            recs.append(f"Decrease allocation to: {', '.join(names)}")

        # Paused
        paused = [r for r in rankings if r.allocation_action == AllocationAction.PAUSE]
        if paused:
            names = [r.strategy_name for r in paused]
            recs.append(f"Consider pausing: {', '.join(names)} (negative expectancy)")

        # Need investigation
        investigate = [r for r in rankings if r.allocation_action == AllocationAction.INVESTIGATE]
        if investigate:
            for r in investigate:
                recs.append(f"Investigate {r.strategy_name}: {r.allocation_reason}")

        # Edge decay warnings
        decay = [r for r in rankings if r.edge_persistence < 0.3 and r.n_trades >= 20]
        for r in decay:
            recs.append(f"⚠ {r.strategy_name} edge may be decaying (persistence: {r.edge_persistence:.0%})")

        return recs

    def _generate_summary(self, rankings: list[StrategyRanking]) -> str:
        """Generate a human-readable summary."""
        if not rankings:
            return "No strategies with completed trades."

        total_trades = sum(r.n_trades for r in rankings)
        total_pnl = sum(r.total_pnl for r in rankings)
        profitable = sum(1 for r in rankings if r.total_pnl > 0)
        avg_composite = np.mean([r.composite_score for r in rankings]) if rankings else 0

        return (
            f"Ranked {len(rankings)} strategies ({total_trades} trades, "
            f"${total_pnl:.0f} total P&L, {profitable} profitable). "
            f"Average composite: {avg_composite:.0f}/100. "
            f"Best: {self._safe_best(rankings)}"
        )

    def _safe_best(self, rankings: list[StrategyRanking]) -> str:
        """Get best strategy name safely."""
        if not rankings:
            return "N/A"
        best = max(rankings, key=lambda r: r.composite_score)
        return f"{best.strategy_name} ({best.composite_score:.0f})"
