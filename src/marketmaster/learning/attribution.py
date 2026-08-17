"""
Signal Attribution Framework — Phase 6 Learning System.

This module tracks which signals, agents, evidence, and strategies contributed to
each trade outcome. It connects entry signals to exit P&L so we can measure edge.

The core idea: every trade has an audit trail. When we enter a position we record
the signal that triggered it, the agent scores that fed the signal, the debate
verdict, and the market regime. When we exit we record the P&L, hold time, and
exit reason. Together these let us answer:

- Which strategies have an edge?
- Which agents' scores actually predict P&L?
- In which market regimes does each strategy work?
- What is our realized expectancy and profit factor?

No external API calls are needed — everything is computed in-process from the
attributions recorded by the tracker.

Usage:
    tracker = AttributionTracker()
    signal_id = tracker.record_entry(signal, fill_price=100.0,
                                     fill_date=datetime(...), regime="RISK_ON")
    tracker.record_exit(signal_id, exit_price=108.0, exit_date=datetime(...),
                        exit_reason=ExitReason.TAKE_PROFIT, regime="RISK_ON")
    stats = tracker.get_strategy_stats("MomentumBreakout")
    print(stats.win_rate, stats.avg_r_multiple, stats.expectancy)
"""

from __future__ import annotations

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class ExitReason(Enum):
    """Why a position was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL_EXIT = "signal_exit"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_price: float,
    direction: str,
) -> float:
    """
    Compute the R-multiple of a trade.

    R-multiple = realized P&L expressed in units of initial risk (the distance
    from entry to the stop). A trade stopped out at exactly the stop is -1R.

    For longs:
        r = (exit_price - entry_price) / (entry_price - stop_price)
    For shorts:
        r = (entry_price - exit_price) / (stop_price - entry_price)

    Args:
        entry_price: Fill price at entry.
        exit_price: Fill price at exit.
        stop_price: Stop price used to define initial risk.
        direction: ``"long"`` or ``"short"`` (case-insensitive).

    Returns:
        The R-multiple as a float. Returns 0.0 when initial risk is zero or the
        direction is unrecognized (so callers never divide by zero).
    """
    direction_norm = (direction or "").lower()
    if direction_norm == "long":
        risk = entry_price - stop_price
        if risk == 0:
            return 0.0
        return (exit_price - entry_price) / risk
    elif direction_norm == "short":
        risk = stop_price - entry_price
        if risk == 0:
            return 0.0
        return (entry_price - exit_price) / risk
    # Neutral / unknown direction: no directional P&L attribution.
    return 0.0


def compute_sharpe(returns: list[float], annualization_factor: float = 252.0) -> float:
    """
    Compute an annualized Sharpe ratio from a list of per-trade (or per-period)
    returns, assuming a zero risk-free rate.

    Sharpe = mean(returns) / std(returns) * sqrt(annualization_factor)

    For a sequence of per-trade returns, ``annualization_factor`` represents the
    approximate number of trades per year (default 252, one trade per trading
    day). Use 1.0 to get a per-trade Sharpe without annualization.

    Args:
        returns: List of fractional returns (e.g. 0.01 = 1%).
        annualization_factor: Number of periods per year for annualization.

    Returns:
        The Sharpe ratio as a float. Returns 0.0 when there are fewer than two
        returns or when the standard deviation is zero (no volatility ⇒ no
        risk-adjusted edge to measure).
    """
    if len(returns) < 2:
        return 0.0
    std = statistics.pstdev(returns)
    if std == 0:
        return 0.0
    mean = statistics.mean(returns)
    return (mean / std) * math.sqrt(annualization_factor)


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """
    Compute the Pearson product-moment correlation between two equal-length lists.

    Returns a value in [-1, 1]. Returns 0.0 when the inputs are empty, of
    mismatched length, or have zero variance (correlation is undefined).
    """
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs = xs[:n]
    ys = ys[:n]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# ─────────────────────────────────────────────────────────────────────────────
# Core attribution dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SignalAttribution:
    """
    Links a single trade signal to its realized outcome.

    Created at entry (``record_entry``) with the signal-side fields populated and
    the exit-side fields left ``None``. Completed at exit (``record_exit``) with
    P&L, hold time, exit reason, and R-multiple filled in.

    Signal-side fields (known at entry):
        signal_id, symbol, strategy_name, regime, signal_date, signal_score,
        signal_confidence, signal_direction, agent_evidence, debate_score,
        debate_winner, entry_price, entry_date, stop_price,
        market_regime_at_entry.

    Exit-side fields (known at exit):
        exit_price, exit_date, pnl_dollars, pnl_pct, hold_days, exit_reason,
        r_multiple, win, market_regime_at_exit.
    """

    # Identity / signal metadata
    signal_id: str
    symbol: str
    strategy_name: str
    regime: str
    signal_date: Optional[datetime] = None

    # Signal quality
    signal_score: float = 0.0
    signal_confidence: float = 0.0
    signal_direction: str = "neutral"

    # Agent evidence: agent_name → that agent's score(s) at signal time.
    # Values may be a single composite score or nested dict; tracker stores
    # whatever the signal carried.
    agent_evidence: dict[str, Any] = field(default_factory=dict)

    # Debate synthesis at signal time
    debate_score: Optional[float] = None
    debate_winner: Optional[str] = None

    # Entry
    entry_price: Optional[float] = None
    entry_date: Optional[datetime] = None
    stop_price: Optional[float] = None

    # Exit (filled in by record_exit)
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    pnl_dollars: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_days: Optional[int] = None
    exit_reason: Optional[ExitReason] = None
    r_multiple: Optional[float] = None
    win: Optional[bool] = None

    # Regime at entry / exit (may differ; regime can change mid-trade)
    market_regime_at_entry: Optional[str] = None
    market_regime_at_exit: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        """True once record_exit has populated the exit fields."""
        return self.exit_price is not None and self.win is not None

    def compute_pnl(
        self,
        exit_price: float,
        shares: float = 1.0,
    ) -> tuple[float, float]:
        """
        Compute dollar and percentage P&L for a given exit price.

        For longs:  pnl = (exit - entry) * shares
        For shorts: pnl = (entry - exit) * shares

        Returns (pnl_dollars, pnl_pct) where pnl_pct is always relative to the
        entry price (sign included).
        """
        if self.entry_price is None or self.entry_price == 0:
            return 0.0, 0.0
        direction = (self.signal_direction or "").lower()
        if direction == "short":
            pnl_dollars = (self.entry_price - exit_price) * shares
        else:  # long (neutral treated as flat, i.e. no P&L)
            pnl_dollars = (exit_price - self.entry_price) * shares
        pnl_pct = pnl_dollars / (self.entry_price * shares) if shares else 0.0
        return pnl_dollars, pnl_pct

    def compute_r_multiple_for_exit(self, exit_price: float) -> float:
        """
        Compute the R-multiple for a candidate exit price using the stored entry
        and stop prices.
        """
        if self.entry_price is None or self.stop_price is None:
            return 0.0
        return compute_r_multiple(
            entry_price=self.entry_price,
            exit_price=exit_price,
            stop_price=self.stop_price,
            direction=self.signal_direction,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Performance stats dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StrategyPerformanceStats:
    """
    Per-strategy realized performance over all closed attributions.

    All figures are computed from closed (exited) trades only; pending/open
    attributions are excluded.
    """

    strategy_name: str
    total_signals: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_r_multiple: float = 0.0
    avg_hold_days: float = 0.0
    best_trade: Optional[float] = None  # best pnl_pct
    worst_trade: Optional[float] = None  # worst pnl_pct
    total_pnl: float = 0.0
    sharpe_estimate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # expected R per trade

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_signals": self.total_signals,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "avg_r_multiple": self.avg_r_multiple,
            "avg_hold_days": self.avg_hold_days,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "total_pnl": self.total_pnl,
            "sharpe_estimate": self.sharpe_estimate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
        }


@dataclass
class AgentPerformanceStats:
    """
    Per-agent contribution tracking.

    Measures whether a given agent's score at signal time is predictive of the
    eventual trade P&L. A strongly positive ``correlation_with_pnl`` means the
    agent adds edge; a near-zero or negative correlation means its score is
    noise (or contrarian) for this universe of trades.
    """

    agent_name: str
    total_signals: int = 0
    avg_agent_score_on_wins: float = 0.0
    avg_agent_score_on_losses: float = 0.0
    correlation_with_pnl: float = 0.0
    top_contributing_markets: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_signals": self.total_signals,
            "avg_agent_score_on_wins": self.avg_agent_score_on_wins,
            "avg_agent_score_on_losses": self.avg_agent_score_on_losses,
            "correlation_with_pnl": self.correlation_with_pnl,
            "top_contributing_markets": self.top_contributing_markets,
        }


@dataclass
class RegimePerformanceStats:
    """
    Per-regime performance summary.

    Identifies which strategies thrive and which struggle in each market regime,
    so the orchestrator can tilt capital toward the right strategies for the
    prevailing environment.
    """

    regime: str
    total_signals: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "total_signals": self.total_signals,
            "win_rate": self.win_rate,
            "avg_pnl_pct": self.avg_pnl_pct,
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AttributionTracker
# ─────────────────────────────────────────────────────────────────────────────


class AttributionTracker:
    """
    Collects and queries signal attributions.

    The tracker is the single source of truth for "which signal became which
    trade and how did it perform". It maintains two views over the same
    attributions:

    - ``_pending``: attributions with an entry but no exit yet, keyed by
      ``signal_id``.
    - ``_closed``: attributions whose exit has been recorded.

    Queries (``get_attributions``, ``get_strategy_stats``, etc.) operate over
    closed attributions by default since P&L is only known post-exit, but
    ``get_attributions`` can optionally include pending ones.

    The tracker is pure-Python and in-memory. Persistence is the caller's
    responsibility (e.g. serialize attributions to the decision log / DB).
    """

    def __init__(self) -> None:
        # signal_id -> attribution
        self._pending: dict[str, SignalAttribution] = {}
        # All closed attributions (insertion-ordered)
        self._closed: list[SignalAttribution] = []
        # signal_id -> attribution (master index, includes pending + closed)
        self._by_id: dict[str, SignalAttribution] = {}

    # ── Recording ─────────────────────────────────────────────────────────

    def record_entry(
        self,
        signal: Any,
        fill_price: float,
        fill_date: datetime,
        regime: str,
        *,
        debate_score: Optional[float] = None,
        debate_winner: Optional[str] = None,
        agent_evidence: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Record a trade entry and create a pending attribution.

        Args:
            signal: A ``TradeSignal`` (or any object with ``symbol``,
                ``strategy_name``, ``direction``, ``score``, ``confidence``,
                ``entry_price``/``stop_price``, ``evidence``, ``regime``, and
                ``as_of`` attributes). Missing attributes default sensibly.
            fill_price: Actual fill price at entry.
            fill_date: Timestamp of the fill.
            regime: Market regime at entry (e.g. ``"RISK_ON"``).
            debate_score: Optional debate net-score at signal time. Falls back
                to the signal's ``debate_score`` attribute if present.
            debate_winner: Optional debate winner (``"bull"``/``"bear"``/
                ``"split"``). Falls back to the signal's ``debate_winner``
                attribute if present.
            agent_evidence: Optional explicit map of agent_name → score(s) at
                signal time. Falls back to the signal's ``evidence`` dict, then
                ``agent_evidence`` attribute, then empty.

        Returns:
            The generated ``signal_id`` for the new pending attribution.
        """
        signal_id = f"sig_{uuid.uuid4().hex[:12]}"

        def _g(attr: str, default: Any = None) -> Any:
            return getattr(signal, attr, default)

        # Direction normalization (handles SignalDirection enum or str)
        raw_direction = _g("direction", "neutral")
        direction_str = (
            raw_direction.value if isinstance(raw_direction, Enum) else str(raw_direction or "neutral")
        ).lower()

        # Agent evidence resolution
        if agent_evidence is not None:
            evidence_map = dict(agent_evidence)
        else:
            evidence_map = _g("agent_evidence", None)
            if evidence_map is None:
                # TradeSignal.evidence is dict[str, float]
                raw_ev = _g("evidence", {}) or {}
                if isinstance(raw_ev, dict):
                    evidence_map = dict(raw_ev)
                else:
                    evidence_map = {}

        # Stop price: prefer explicit, then signal's stop_price
        stop_price = _g("stop_price", None)

        attribution = SignalAttribution(
            signal_id=signal_id,
            symbol=_g("symbol", ""),
            strategy_name=_g("strategy_name", ""),
            regime=regime,
            signal_date=_g("as_of", None),
            signal_score=float(_g("score", 0.0) or 0.0),
            signal_confidence=float(_g("confidence", 0.0) or 0.0),
            signal_direction=direction_str,
            agent_evidence=evidence_map,
            debate_score=debate_score if debate_score is not None else _g("debate_score", None),
            debate_winner=debate_winner if debate_winner is not None else _g("debate_winner", None),
            entry_price=fill_price,
            entry_date=fill_date,
            stop_price=stop_price,
            market_regime_at_entry=regime,
        )

        self._pending[signal_id] = attribution
        self._by_id[signal_id] = attribution
        return signal_id

    def record_exit(
        self,
        signal_id: str,
        exit_price: float,
        exit_date: datetime,
        exit_reason: ExitReason,
        regime: Optional[str] = None,
        *,
        shares: float = 1.0,
    ) -> Optional[SignalAttribution]:
        """
        Record a trade exit and complete the pending attribution with P&L.

        Args:
            signal_id: The signal_id returned by ``record_entry``.
            exit_price: Actual fill price at exit.
            exit_date: Timestamp of the exit fill.
            exit_reason: Why the position was closed.
            regime: Market regime at exit (may differ from entry regime).
            shares: Number of shares closed, used for dollar P&L. pnl_pct is
                always per-unit and independent of shares.

        Returns:
            The completed ``SignalAttribution``, or ``None`` if ``signal_id`` is
            unknown or already closed.
        """
        attribution = self._by_id.get(signal_id)
        if attribution is None or attribution.is_closed:
            return None

        pnl_dollars, pnl_pct = attribution.compute_pnl(exit_price, shares=shares)
        r_multiple = attribution.compute_r_multiple_for_exit(exit_price)

        # Hold time in whole days (floor of the timedelta).
        hold_days: Optional[int] = None
        if attribution.entry_date is not None:
            delta = exit_date - attribution.entry_date
            hold_days = max(0, int(delta.total_seconds() // 86400))

        attribution.exit_price = exit_price
        attribution.exit_date = exit_date
        attribution.pnl_dollars = pnl_dollars
        attribution.pnl_pct = pnl_pct
        attribution.hold_days = hold_days
        attribution.exit_reason = exit_reason
        attribution.r_multiple = r_multiple
        attribution.win = pnl_dollars > 0
        attribution.market_regime_at_exit = regime

        # Move from pending to closed.
        self._pending.pop(signal_id, None)
        self._closed.append(attribution)
        return attribution

    # ── Querying ───────────────────────────────────────────────────────────

    def get_attributions(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        date_range: Optional[tuple[datetime, datetime]] = None,
        wins_only: bool = False,
        losses_only: bool = False,
        include_pending: bool = False,
    ) -> list[SignalAttribution]:
        """
        Query attributions with optional filters.

        Filters are AND-combined. By default only closed attributions are
        returned (P&L is unknown for pending trades). Set ``include_pending=True``
        to also return open attributions (their exit/P&L fields are ``None``).

        Args:
            symbol: Filter by ticker.
            strategy: Filter by strategy name.
            regime: Filter by regime at entry.
            date_range: ``(start, end)`` inclusive; filters on ``entry_date``.
            wins_only: Return only winning trades (pnl_dollars > 0).
            losses_only: Return only losing trades (pnl_dollars <= 0).
            include_pending: Also include open/pending attributions.

        Returns:
            List of matching ``SignalAttribution`` objects (oldest first).
        """
        if wins_only and losses_only:
            # Contradictory filters — return nothing.
            return []

        results: list[SignalAttribution] = list(self._closed)
        if include_pending:
            results = results + list(self._pending.values())

        def _matches(a: SignalAttribution) -> bool:
            if symbol is not None and a.symbol != symbol:
                return False
            if strategy is not None and a.strategy_name != strategy:
                return False
            if regime is not None and a.regime != regime:
                return False
            if date_range is not None:
                start, end = date_range
                if a.entry_date is not None:
                    if a.entry_date < start or a.entry_date > end:
                        return False
                else:
                    # No entry date means we cannot place it in the range.
                    return False
            if wins_only and not (a.pnl_dollars is not None and a.pnl_dollars > 0):
                return False
            if losses_only and not (a.pnl_dollars is not None and a.pnl_dollars <= 0):
                return False
            return True

        return [a for a in results if _matches(a)]

    def get_strategy_stats(self, strategy_name: str) -> StrategyPerformanceStats:
        """
        Compute realized performance stats for a single strategy.

        Only closed trades for ``strategy_name`` are considered. If there are no
        closed trades, a zeroed stats object is returned (so callers can always
        serialize it safely).
        """
        trades = [
            a for a in self._closed
            if a.strategy_name == strategy_name
        ]
        return self._compute_strategy_stats(strategy_name, trades)

    def get_agent_stats(self, agent_name: str) -> AgentPerformanceStats:
        """
        Compute contribution stats for a single agent.

        For every closed attribution that carried this agent in its
        ``agent_evidence``, we extract the agent's score and correlate it with
        the realized pnl_pct. This answers: "does this agent's score predict
        P&L?"

        Agents that never appear in any closed attribution return a zeroed
        stats object.
        """
        agent_scores: list[float] = []
        pnl_pcts: list[float] = []
        win_scores: list[float] = []
        loss_scores: list[float] = []
        market_counts: dict[str, int] = {}

        for a in self._closed:
            if agent_name not in a.agent_evidence:
                continue
            raw_score = a.agent_evidence[agent_name]
            score = self._scalar_score(raw_score)
            if score is None:
                continue
            pnl = a.pnl_pct if a.pnl_pct is not None else 0.0
            agent_scores.append(score)
            pnl_pcts.append(pnl)
            if a.pnl_dollars is not None and a.pnl_dollars > 0:
                win_scores.append(score)
            else:
                loss_scores.append(score)
            market_counts[a.symbol] = market_counts.get(a.symbol, 0) + 1

        total = len(agent_scores)
        top_markets = sorted(market_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return AgentPerformanceStats(
            agent_name=agent_name,
            total_signals=total,
            avg_agent_score_on_wins=statistics.mean(win_scores) if win_scores else 0.0,
            avg_agent_score_on_losses=statistics.mean(loss_scores) if loss_scores else 0.0,
            correlation_with_pnl=_pearson_correlation(agent_scores, pnl_pcts),
            top_contributing_markets=top_markets,
        )

    def get_regime_stats(self, regime: str) -> RegimePerformanceStats:
        """
        Compute performance stats for a single market regime.

        Regime is matched on the entry regime (``market_regime_at_entry``).
        Identifies the best- and worst-performing strategies within the regime
        by average pnl_pct.
        """
        trades = [
            a for a in self._closed
            if (a.market_regime_at_entry or a.regime) == regime
        ]

        if not trades:
            return RegimePerformanceStats(regime=regime)

        wins = sum(1 for a in trades if a.pnl_dollars is not None and a.pnl_dollars > 0)
        pnl_pcts = [a.pnl_pct or 0.0 for a in trades if a.pnl_pct is not None]

        # Per-strategy avg pnl_pct within this regime
        strat_pnl: dict[str, list[float]] = {}
        for a in trades:
            strat_pnl.setdefault(a.strategy_name, []).append(a.pnl_pct or 0.0)
        strat_avg = {s: statistics.mean(v) for s, v in strat_pnl.items()}
        best_strategy = max(strat_avg, key=strat_avg.get) if strat_avg else None
        worst_strategy = min(strat_avg, key=strat_avg.get) if strat_avg else None

        return RegimePerformanceStats(
            regime=regime,
            total_signals=len(trades),
            win_rate=wins / len(trades),
            avg_pnl_pct=statistics.mean(pnl_pcts) if pnl_pcts else 0.0,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
        )

    def summary(self) -> dict[str, Any]:
        """
        Return a high-level summary of overall performance.

        Includes counts (pending/closed/total), win rate, average P&L and
        R-multiple, total P&L, profit factor, expectancy, and a Sharpe
        estimate across all closed trades. Also lists the distinct strategies,
        agents, and regimes observed.
        """
        closed = self._closed
        n_closed = len(closed)
        n_pending = len(self._pending)

        if n_closed == 0:
            return {
                "total_signals": n_pending,
                "closed_trades": 0,
                "pending_trades": n_pending,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "avg_r_multiple": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "sharpe_estimate": 0.0,
                "strategies": [],
                "agents": [],
                "regimes": [],
            }

        wins = sum(1 for a in closed if a.pnl_dollars is not None and a.pnl_dollars > 0)
        pnl_pcts = [a.pnl_pct or 0.0 for a in closed if a.pnl_pct is not None]
        r_multiples = [a.r_multiple or 0.0 for a in closed if a.r_multiple is not None]
        gross_profit = sum(a.pnl_dollars for a in closed if a.pnl_dollars and a.pnl_dollars > 0)
        gross_loss = abs(sum(a.pnl_dollars for a in closed if a.pnl_dollars and a.pnl_dollars < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        strategies = sorted({a.strategy_name for a in closed if a.strategy_name})
        agents = sorted(
            {name for a in closed for name in a.agent_evidence.keys()}
        )
        regimes = sorted({
            (a.market_regime_at_entry or a.regime) for a in closed
        })

        return {
            "total_signals": n_closed + n_pending,
            "closed_trades": n_closed,
            "pending_trades": n_pending,
            "win_rate": wins / n_closed,
            "avg_pnl_pct": statistics.mean(pnl_pcts) if pnl_pcts else 0.0,
            "avg_r_multiple": statistics.mean(r_multiples) if r_multiples else 0.0,
            "total_pnl": sum(a.pnl_dollars or 0.0 for a in closed),
            "profit_factor": profit_factor,
            "expectancy": statistics.mean(r_multiples) if r_multiples else 0.0,
            "sharpe_estimate": compute_sharpe(pnl_pcts),
            "strategies": strategies,
            "agents": agents,
            "regimes": regimes,
        }

    # ── Introspection helpers ─────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of open attributions awaiting an exit."""
        return len(self._pending)

    @property
    def closed_count(self) -> int:
        """Number of completed attributions."""
        return len(self._closed)

    def get_pending(self) -> list[SignalAttribution]:
        """Return all open attributions (no exit recorded yet)."""
        return list(self._pending.values())

    def get_attribution(self, signal_id: str) -> Optional[SignalAttribution]:
        """Look up a single attribution by signal_id (pending or closed)."""
        return self._by_id.get(signal_id)

    def all_strategies(self) -> list[str]:
        """Distinct strategy names across all recorded attributions."""
        return sorted({a.strategy_name for a in self._closed + list(self._pending.values()) if a.strategy_name})

    def all_agents(self) -> list[str]:
        """Distinct agent names that appear in any recorded attribution's evidence."""
        names: set[str] = set()
        for a in self._closed + list(self._pending.values()):
            names.update(a.agent_evidence.keys())
        return sorted(names)

    def all_regimes(self) -> list[str]:
        """Distinct regimes observed across all recorded attributions."""
        regimes: set[str] = set()
        for a in self._closed + list(self._pending.values()):
            regimes.add(a.regime)
            if a.market_regime_at_entry:
                regimes.add(a.market_regime_at_entry)
            if a.market_regime_at_exit:
                regimes.add(a.market_regime_at_exit)
        return sorted(r for r in regimes if r)

    # ── Internal computation helpers ───────────────────────────────────────

    @staticmethod
    def _scalar_score(raw: Any) -> Optional[float]:
        """
        Reduce an agent evidence value to a single scalar score.

        Agent evidence values may be a plain float, a dict of sub-scores, or a
        list. We take the mean of any dict values / list elements; floats pass
        through. ``None``/empty returns ``None`` (agent score unusable).
        """
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, dict):
            vals = [v for v in raw.values() if isinstance(v, (int, float))]
            if not vals:
                return None
            return statistics.mean(vals)
        if isinstance(raw, (list, tuple)):
            vals = [v for v in raw if isinstance(v, (int, float))]
            if not vals:
                return None
            return statistics.mean(vals)
        return None

    @staticmethod
    def _compute_strategy_stats(
        strategy_name: str,
        trades: list[SignalAttribution],
    ) -> StrategyPerformanceStats:
        """Compute StrategyPerformanceStats from a list of closed trades."""
        if not trades:
            return StrategyPerformanceStats(strategy_name=strategy_name)

        wins = [a for a in trades if a.pnl_dollars is not None and a.pnl_dollars > 0]
        pnl_pcts = [a.pnl_pct or 0.0 for a in trades if a.pnl_pct is not None]
        r_multiples = [a.r_multiple or 0.0 for a in trades if a.r_multiple is not None]
        hold_days = [a.hold_days for a in trades if a.hold_days is not None]
        total_pnl = sum(a.pnl_dollars or 0.0 for a in trades)

        gross_profit = sum(a.pnl_dollars for a in trades if a.pnl_dollars and a.pnl_dollars > 0)
        gross_loss = abs(sum(a.pnl_dollars for a in trades if a.pnl_dollars and a.pnl_dollars < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        best_trade = max(pnl_pcts) if pnl_pcts else None
        worst_trade = min(pnl_pcts) if pnl_pcts else None

        return StrategyPerformanceStats(
            strategy_name=strategy_name,
            total_signals=len(trades),
            win_rate=len(wins) / len(trades),
            avg_pnl_pct=statistics.mean(pnl_pcts) if pnl_pcts else 0.0,
            avg_r_multiple=statistics.mean(r_multiples) if r_multiples else 0.0,
            avg_hold_days=statistics.mean(hold_days) if hold_days else 0.0,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_pnl=total_pnl,
            sharpe_estimate=compute_sharpe(pnl_pcts),
            profit_factor=profit_factor,
            expectancy=statistics.mean(r_multiples) if r_multiples else 0.0,
        )
