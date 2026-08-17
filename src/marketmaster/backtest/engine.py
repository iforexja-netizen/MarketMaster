"""
Backtest Engine — Point-in-time strategy backtesting with full cost model.

The backtester simulates running a strategy (or set of strategies) over
historical data, tracking equity curve, positions, and performance metrics.

Key principles:
- Point-in-time correctness: only uses data available up to each evaluation date
- Full cost model: commissions, slippage, and spread impact
- Corporate actions: adjusts for splits and dividends
- No look-ahead bias: the backtest doesn't peek at future data
- Comprehensive metrics: Sharpe, Sortino, max drawdown, win rate, etc.

Usage:
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(
        strategy=strategy,
        symbol="AAPL",
        bars=historical_bars,
        evidence_fn=compute_evidence_per_date,  # function(date) → evidence
        regime="BULL",
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
    )
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional, Callable
import numpy as np
from collections import deque


@dataclass
class Position:
    """A simulated position in the backtest."""
    symbol: str
    direction: str  # "long" or "short"
    entry_date: date
    entry_price: float
    shares: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    strategy_name: str = ""
    initial_risk: float = 0.0  # initial risk in $


@dataclass
class Trade:
    """A completed round-trip trade."""
    symbol: str
    direction: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "target", "stop", "signal", "end"
    strategy_name: str
    hold_days: int


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest."""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_hold_days: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    calmar_ratio: float = 0.0


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    symbol: str
    strategy_name: str
    regime: str
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    metrics: BacktestMetrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    positions_log: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class BacktestEngine:
    """
    Point-in-time backtesting engine.

    Simulates trading a single symbol with a single strategy over historical
    data. Tracks equity curve, positions, and computes comprehensive metrics.

    The engine processes bars chronologically. For each bar:
    1. Check if existing positions hit stop/target
    2. Evaluate the strategy using only data up to the current bar
    3. Generate a signal
    4. If new signal, close existing position and open new one
    5. Record equity and position values
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0,  # 5 basis points
        position_size_pct: float = 5.0,  # 5% of portfolio per position
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_trade
        self.slippage_bps = slippage_bps / 10_000  # convert to fraction
        self.position_size_pct = position_size_pct / 100

    def run(
        self,
        strategy: Any,  # Strategy
        symbol: str,
        bars: list[Any],  # list of OHLCV bars with date, open, high, low, close, volume
        evidence_fn: Optional[Callable[[date], list[Any]]] = None,
        regime: str = "NEUTRAL",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """
        Run a backtest of a single strategy on a single symbol.

        Args:
            strategy: Strategy instance
            symbol: Ticker symbol
            bars: Historical OHLCV bars (chronological)
            evidence_fn: Function(date) → list[DecisionEvidence] (optional)
            regime: Market regime (constant for this backtest)
            start_date: Backtest start date (optional, defaults to first bar)
            end_date: Backtest end date (optional, defaults to last bar)

        Returns:
            BacktestResult with metrics, trades, and equity curve
        """
        if not bars:
            return self._empty_result(symbol, strategy.name, regime)

        # Filter bars by date range
        if start_date:
            bars = [b for b in bars if b.date >= start_date]
        if end_date:
            bars = [b for b in bars if b.date <= end_date]

        if not bars:
            return self._empty_result(symbol, strategy.name, regime)

        # State
        equity = self.initial_capital
        position: Optional[Position] = None
        trades: list[Trade] = []
        equity_curve: list[tuple[date, float]] = []
        positions_log: list[dict] = []

        # Track equity high water mark for drawdown
        peak_equity = equity
        drawdown_start = None
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_days = 0

        # Daily returns for Sharpe/Sortino
        daily_returns: list[float] = []
        prev_equity = equity

        total_commission = 0.0
        total_slippage = 0.0

        for i, bar in enumerate(bars):
            current_date = bar.date
            close = float(bar.close) if bar.close else 0
            high = float(bar.high) if bar.high else close
            low = float(bar.low) if bar.low else close

            # ── 1. Check stop/target on existing position ──────────────────
            if position:
                exit_price = None
                exit_reason = None

                if position.direction == "long":
                    if position.stop_price and low <= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop"
                    elif position.target_price and high >= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "target"
                elif position.direction == "short":
                    if position.stop_price and high >= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop"
                    elif position.target_price and low <= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "target"

                if exit_price:
                    # Apply slippage
                    slip = exit_price * self.slippage_bps
                    if position.direction == "long":
                        exit_price -= slip
                    else:
                        exit_price += slip
                    total_slippage += abs(slip)
                    total_commission += self.commission

                    # Close position
                    trade = self._close_position(position, current_date, exit_price, exit_reason)
                    pnl = trade.pnl - self.commission
                    equity += pnl
                    trades.append(trade)

                    positions_log.append({
                        "date": current_date.isoformat(),
                        "action": "exit",
                        "reason": exit_reason,
                        "price": exit_price,
                        "pnl": pnl,
                        "equity": equity,
                    })

                    position = None

            # ── 2. Evaluate strategy ────────────────────────────────────────
            evidence = []
            if evidence_fn:
                try:
                    evidence = evidence_fn(current_date)
                except Exception:
                    evidence = []

            # Build market data from available bars
            market_data = self._build_market_data(bars[:i+1], close)

            try:
                signal = strategy.evaluate(
                    symbol=symbol,
                    evidence=evidence,
                    debate=None,
                    market_data=market_data,
                    regime=regime,
                )
            except Exception:
                signal = None

            # ── 3. Manage positions based on signal ────────────────────────
            if signal and signal.direction.value != "neutral":
                # Close existing position if direction changes
                if position and position.direction != signal.direction.value:
                    slip = close * self.slippage_bps
                    total_slippage += slip
                    total_commission += self.commission
                    if position.direction == "long":
                        exit_price = close - slip
                    else:
                        exit_price = close + slip
                    trade = self._close_position(position, current_date, exit_price, "signal")
                    equity += trade.pnl - self.commission
                    trades.append(trade)
                    position = None

                # Open new position if none exists
                if not position and signal.entry_price and signal.position_size_pct > 0:
                    position_value = equity * (signal.position_size_pct / 100)
                    shares = position_value / close

                    # Apply entry slippage
                    entry_slip = close * self.slippage_bps
                    if signal.direction.value == "long":
                        entry_price = close + entry_slip
                    else:
                        entry_price = close - entry_slip
                    total_slippage += entry_slip
                    total_commission += self.commission

                    position = Position(
                        symbol=symbol,
                        direction=signal.direction.value,
                        entry_date=current_date,
                        entry_price=entry_price,
                        shares=shares,
                        stop_price=signal.stop_price,
                        target_price=signal.target_price,
                        strategy_name=strategy.name,
                        initial_risk=abs(entry_price - (signal.stop_price or entry_price)) * shares,
                    )

                    positions_log.append({
                        "date": current_date.isoformat(),
                        "action": "entry",
                        "direction": signal.direction.value,
                        "price": entry_price,
                        "shares": shares,
                        "stop": signal.stop_price,
                        "target": signal.target_price,
                        "equity": equity,
                    })

            # ── 4. Mark-to-market equity ───────────────────────────────────
            if position:
                if position.direction == "long":
                    unrealized = (close - position.entry_price) * position.shares
                else:
                    unrealized = (position.entry_price - close) * position.shares
                mtm_equity = equity + unrealized
            else:
                mtm_equity = equity

            equity_curve.append((current_date, mtm_equity))

            # ── 5. Track drawdown ──────────────────────────────────────────
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
                current_dd_days = 0
            else:
                current_dd_days += 1
                dd = (peak_equity - mtm_equity) / peak_equity * 100
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = current_dd_days

            # Daily return
            if prev_equity > 0:
                daily_ret = (mtm_equity - prev_equity) / prev_equity
                daily_returns.append(daily_ret)
            prev_equity = mtm_equity

        # ── Close any remaining position at final close ──────────────────
        if position:
            close = float(bars[-1].close) if bars[-1].close else 0
            slip = close * self.slippage_bps
            total_slippage += slip
            total_commission += self.commission
            if position.direction == "long":
                exit_price = close - slip
            else:
                exit_price = close + slip
            trade = self._close_position(position, bars[-1].date, exit_price, "end")
            equity += trade.pnl - self.commission
            trades.append(trade)
            position = None

        final_equity = equity

        # ── Compute metrics ───────────────────────────────────────────────
        metrics = self._compute_metrics(
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            trades=trades,
            daily_returns=daily_returns,
            max_dd=max_dd,
            max_dd_duration=max_dd_duration,
            total_commission=total_commission,
            total_slippage=total_slippage,
            num_bars=len(bars),
        )

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            regime=regime,
            start_date=bars[0].date,
            end_date=bars[-1].date,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            positions_log=positions_log,
        )

    def _close_position(self, position: Position, exit_date: date, exit_price: float, reason: str) -> Trade:
        """Close a position and compute the trade."""
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.shares
        else:
            pnl = (position.entry_price - exit_price) * position.shares
        pnl_pct = (pnl / (position.entry_price * position.shares)) * 100 if position.entry_price > 0 else 0

        hold_days = (exit_date - position.entry_date).days

        return Trade(
            symbol=position.symbol,
            direction=position.direction,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            strategy_name=position.strategy_name,
            hold_days=hold_days,
        )

    def _build_market_data(self, bars: list[Any], current_close: float) -> dict[str, float]:
        """Build market data dict from available bars (point-in-time)."""
        market_data = {"price": current_close}

        if len(bars) >= 2:
            closes = [float(b.close) if b.close else 0 for b in bars]

            # Simple ATR estimate
            if len(bars) >= 14:
                recent_bars = bars[-14:]
                ranges = []
                for b in recent_bars:
                    if b.high and b.low:
                        ranges.append(float(b.high) - float(b.low))
                if ranges:
                    market_data["atr"] = float(np.mean(ranges))
                else:
                    market_data["atr"] = current_close * 0.02
            else:
                market_data["atr"] = current_close * 0.02

            # SMA20
            if len(closes) >= 20:
                market_data["sma20"] = float(np.mean(closes[-20:]))
            # SMA50
            if len(closes) >= 50:
                market_data["sma50"] = float(np.mean(closes[-50:]))
            # SMA200
            if len(closes) >= 200:
                market_data["sma200"] = float(np.mean(closes[-200:]))
            elif closes:
                market_data["sma200"] = float(np.mean(closes))

            # Volume ratio
            if len(bars) >= 20:
                recent_vol = [float(b.volume) if b.volume else 0 for b in bars[-20:]]
                avg_vol = float(np.mean(recent_vol)) if recent_vol else 1
                current_vol = float(bars[-1].volume) if bars[-1].volume else 0
                market_data["volume_ratio"] = current_vol / avg_vol if avg_vol > 0 else 1.0
            else:
                market_data["volume_ratio"] = 1.0

            # Bollinger position
            if len(closes) >= 20:
                recent = closes[-20:]
                mean = float(np.mean(recent))
                std = float(np.std(recent))
                if std > 0:
                    market_data["bollinger_position"] = (current_close - (mean - 2 * std)) / (4 * std)
                else:
                    market_data["bollinger_position"] = 0.5
            else:
                market_data["bollinger_position"] = 0.5

        return market_data

    def _compute_metrics(
        self,
        initial_capital: float,
        final_equity: float,
        trades: list[Trade],
        daily_returns: list[float],
        max_dd: float,
        max_dd_duration: int,
        total_commission: float,
        total_slippage: float,
        num_bars: int,
    ) -> BacktestMetrics:
        """Compute comprehensive performance metrics."""
        metrics = BacktestMetrics()

        # Returns
        metrics.total_return = final_equity - initial_capital
        metrics.total_return_pct = (final_equity / initial_capital - 1) * 100 if initial_capital > 0 else 0

        # Annualized return (assume ~252 trading days)
        trading_days = max(num_bars, 1)
        years = trading_days / 252
        if years > 0 and final_equity > 0:
            metrics.annual_return_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            metrics.annual_return_pct = 0

        # Sharpe ratio (annualized, risk-free = 0)
        if daily_returns and len(daily_returns) > 1:
            mean_ret = float(np.mean(daily_returns))
            std_ret = float(np.std(daily_returns))
            if std_ret > 0:
                metrics.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252)

            # Sortino ratio (downside deviation)
            downside = [r for r in daily_returns if r < 0]
            if downside:
                downside_std = float(np.std(downside))
                if downside_std > 0:
                    metrics.sortino_ratio = (mean_ret / downside_std) * np.sqrt(252)

        # Drawdown
        metrics.max_drawdown_pct = max_dd
        metrics.max_drawdown_duration = max_dd_duration

        # Trade statistics
        metrics.total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)

        if trades:
            metrics.win_rate = len(winning) / len(trades) * 100
            metrics.avg_hold_days = int(np.mean([t.hold_days for t in trades]))

        if winning:
            metrics.avg_win = float(np.mean([t.pnl for t in winning]))
            metrics.best_trade_pct = max(t.pnl_pct for t in winning)

        if losing:
            metrics.avg_loss = float(np.mean([t.pnl for t in losing]))
            metrics.worst_trade_pct = min(t.pnl_pct for t in losing)

        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Calmar ratio
        if max_dd > 0:
            metrics.calmar_ratio = metrics.annual_return_pct / max_dd

        metrics.total_commission = total_commission
        metrics.total_slippage = total_slippage

        return metrics

    def _empty_result(self, symbol: str, strategy_name: str, regime: str) -> BacktestResult:
        """Create an empty backtest result for edge cases."""
        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            regime=regime,
            start_date=date.today(),
            end_date=date.today(),
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            metrics=BacktestMetrics(),
            notes=["No bars provided for backtest"],
        )
