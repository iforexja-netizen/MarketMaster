"""
MarketMaster API Routes — Phase 4: Strategy Plane

Endpoints for strategy registry, screening, backtesting, walk-forward
validation, portfolio optimization, and trade construction.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from marketmaster.db.session import get_db

phase4_router = APIRouter()


# ============================================================================
# Strategy Registry
# ============================================================================

@phase4_router.get("/strategies")
def list_strategies(regime: Optional[str] = None):
    """
    List all available strategies, optionally filtered by regime.
    """
    from marketmaster.strategies.registry import StrategyRegistry
    registry = StrategyRegistry()

    if regime:
        strategies = registry.for_regime(regime)
    else:
        strategies = registry.all()

    return {
        "total": len(strategies),
        "regime": regime,
        "strategies": [
            {
                "name": s.name,
                "description": s.description,
                "applicable_regimes": s.applicable_regimes,
                "max_position_pct": s.config.max_position_pct,
            }
            for s in strategies
        ],
    }


@phase4_router.get("/strategies/{strategy_name}")
def get_strategy(strategy_name: str):
    """Get details of a specific strategy."""
    from marketmaster.strategies.registry import StrategyRegistry
    registry = StrategyRegistry()

    try:
        s = registry.get(strategy_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

    return {
        "name": s.name,
        "description": s.description,
        "applicable_regimes": s.applicable_regimes,
        "config": {
            "max_position_pct": s.config.max_position_pct,
            "stop_loss_pct": s.config.stop_loss_pct,
            "take_profit_pct": s.config.take_profit_pct,
            "max_positions": s.config.max_positions,
            "min_score": s.config.min_score,
            "min_confidence": s.config.min_confidence,
        },
    }


# ============================================================================
# Screener
# ============================================================================

class ScreenRequest(BaseModel):
    universe: list[str]
    regime: str = "NEUTRAL"
    top_n: int = 20
    min_score: float = 55.0
    min_confidence: float = 0.25
    as_of: Optional[date] = None


@phase4_router.post("/screen")
def screen_universe(req: ScreenRequest, db: Session = Depends(get_db)):
    """
    Screen the security universe for trade opportunities.

    Returns ranked trade signals from all applicable strategies.
    """
    from marketmaster.strategies.screener import Screener
    from marketmaster.agents.orchestrator import MarketMasterOrchestrator

    if req.as_of is None:
        req.as_of = date.today()

    screener = Screener()
    orch = MarketMasterOrchestrator(db_session=db)

    result = screener.scan(
        universe=req.universe,
        regime=req.regime,
        as_of=req.as_of,
        analyze_fn=orch.analyze,
        top_n=req.top_n,
        min_score=req.min_score,
        min_confidence=req.min_confidence,
    )

    return {
        "as_of": req.as_of.isoformat(),
        "regime": result.regime,
        "universe_size": result.universe_size,
        "screened": result.screened,
        "active_strategies": result.active_strategies,
        "total_signals": len(result.signals),
        "top_opportunities": [
            {
                "symbol": s.symbol,
                "strategy": s.strategy_name,
                "direction": s.direction.value,
                "score": s.score,
                "confidence": s.confidence,
                "entry_price": s.entry_price,
                "target_price": s.target_price,
                "stop_price": s.stop_price,
                "position_size_pct": s.position_size_pct,
                "risk_reward_ratio": s.risk_reward_ratio,
                "reasoning": s.reasoning,
            }
            for s in result.top_opportunities
        ],
        "errors": result.errors[:10],
    }


# ============================================================================
# Backtester
# ============================================================================

class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    regime: str = "NEUTRAL"
    initial_capital: float = 100_000
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@phase4_router.post("/backtest")
def run_backtest(req: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run a point-in-time backtest of a strategy on a single symbol.

    Returns performance metrics, trade history, and equity curve.
    """
    from marketmaster.backtest.engine import BacktestEngine
    from marketmaster.strategies.registry import StrategyRegistry
    from marketmaster.data.plane import DataPlane

    registry = StrategyRegistry()

    try:
        strategy = registry.get(req.strategy_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {req.strategy_name}")

    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(req.symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {req.symbol}")

    start = req.start_date or date(2020, 1, 1)
    end = req.end_date or date.today()
    bars = plane.get_ohlcv_daily(sec.id, start_date=start, end_date=end)

    if not bars:
        raise HTTPException(status_code=422, detail="No historical bars available for backtest")

    engine = BacktestEngine(initial_capital=req.initial_capital)
    result = engine.run(
        strategy=strategy,
        symbol=req.symbol,
        bars=bars,
        regime=req.regime,
        start_date=start,
        end_date=end,
    )

    return {
        "symbol": result.symbol,
        "strategy": result.strategy_name,
        "regime": result.regime,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": result.initial_capital,
        "final_equity": result.final_equity,
        "metrics": {
            "total_return_pct": result.metrics.total_return_pct,
            "annual_return_pct": result.metrics.annual_return_pct,
            "sharpe_ratio": result.metrics.sharpe_ratio,
            "sortino_ratio": result.metrics.sortino_ratio,
            "max_drawdown_pct": result.metrics.max_drawdown_pct,
            "win_rate": result.metrics.win_rate,
            "total_trades": result.metrics.total_trades,
            "profit_factor": result.metrics.profit_factor,
            "avg_hold_days": result.metrics.avg_hold_days,
            "avg_win": result.metrics.avg_win,
            "avg_loss": result.metrics.avg_loss,
        },
        "trades": [
            {
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "hold_days": t.hold_days,
            }
            for t in result.trades
        ],
        "equity_curve": [
            {"date": d.isoformat(), "equity": e}
            for d, e in result.equity_curve[-100:]  # Last 100 points
        ],
    }


# ============================================================================
# Walk-Forward Validation
# ============================================================================

class WalkForwardRequest(BaseModel):
    strategy_name: str
    symbol: str
    regime: str = "NEUTRAL"
    train_days: int = 504
    test_days: int = 126
    step_days: int = 63


@phase4_router.post("/walkforward")
def run_walkforward(req: WalkForwardRequest, db: Session = Depends(get_db)):
    """
    Run walk-forward validation to test for strategy overfitting.

    Returns IS/OOS performance per window and WFE ratio.
    """
    from marketmaster.backtest.walk_forward import WalkForwardValidator
    from marketmaster.strategies.registry import StrategyRegistry
    from marketmaster.data.plane import DataPlane

    registry = StrategyRegistry()
    try:
        strategy = registry.get(req.strategy_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {req.strategy_name}")

    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(req.symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {req.symbol}")

    bars = plane.get_ohlcv_daily(sec.id, start_date=date(2010, 1, 1), end_date=date.today())

    validator = WalkForwardValidator(
        train_days=req.train_days,
        test_days=req.test_days,
        step_days=req.step_days,
    )
    result = validator.run(
        strategy=strategy,
        symbol=req.symbol,
        bars=bars,
        regime=req.regime,
    )

    return {
        "symbol": result.symbol,
        "strategy": result.strategy_name,
        "windows": len(result.windows),
        "avg_is_return": result.avg_is_return,
        "avg_oos_return": result.avg_oos_return,
        "wfe_ratio": result.wfe_ratio,
        "oos_consistency": result.consistency,
        "avg_oos_sharpe": result.avg_oos_sharpe,
        "avg_oos_max_dd": result.avg_oos_max_dd,
        "overfit_warning": result.overfit_warning,
        "summary": result.summary,
    }


# ============================================================================
# Portfolio Optimizer
# ============================================================================

class OptimizeRequest(BaseModel):
    signals: list[dict]  # Serialized TradeSignal objects
    method: str = "score_weighted"
    max_positions: int = 10
    max_position_pct: float = 5.0
    min_cash_pct: float = 10.0
    regime: str = "NEUTRAL"
    initial_capital: float = 100_000


@phase4_router.post("/portfolio/optimize")
def optimize_portfolio(req: OptimizeRequest):
    """
    Optimize capital allocation across trade signals.

    Methods: equal_weight, score_weighted, risk_parity, mean_variance.
    """
    from marketmaster.portfolio.optimizer import PortfolioOptimizer
    from marketmaster.strategies.base import TradeSignal, SignalDirection

    # Reconstruct TradeSignal objects from dicts
    signals = []
    for s in req.signals:
        signals.append(TradeSignal(
            symbol=s["symbol"],
            strategy_name=s["strategy_name"],
            direction=SignalDirection(s.get("direction", "long")),
            score=s.get("score", 50),
            confidence=s.get("confidence", 0.5),
            entry_price=s.get("entry_price"),
            target_price=s.get("target_price"),
            stop_price=s.get("stop_price"),
            position_size_pct=s.get("position_size_pct", 5),
            risk_reward_ratio=s.get("risk_reward_ratio", 0),
        ))

    optimizer = PortfolioOptimizer(initial_capital=req.initial_capital)
    allocation = optimizer.optimize(
        signals=signals,
        method=req.method,
        max_positions=req.max_positions,
        max_position_pct=req.max_position_pct,
        min_cash_pct=req.min_cash_pct,
        regime=req.regime,
    )

    return {
        "as_of": allocation.as_of.isoformat(),
        "method": allocation.method,
        "n_positions": allocation.n_positions,
        "total_allocation": allocation.total_allocation,
        "cash_reserve": allocation.cash_reserve,
        "avg_score": allocation.avg_score,
        "avg_confidence": allocation.avg_confidence,
        "positions": [
            {
                "symbol": p.symbol,
                "strategy": p.strategy_name,
                "direction": p.direction,
                "weight": p.weight,
                "dollar_allocation": p.dollar_allocation,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
                "target_price": p.target_price,
                "risk_reward": p.risk_reward_ratio,
                "score": p.score,
                "confidence": p.confidence,
            }
            for p in allocation.positions
        ],
        "notes": allocation.notes,
    }


# ============================================================================
# Trade Construction
# ============================================================================

class ConstructRequest(BaseModel):
    allocation: dict  # Serialized PortfolioAllocation
    use_limit_orders: bool = True


@phase4_router.post("/portfolio/construct")
def construct_trades(req: ConstructRequest):
    """
    Construct concrete order instructions from a portfolio allocation.

    Generates entry, stop loss, and take profit orders.
    """
    from marketmaster.portfolio.construction import TradeConstructor
    from marketmaster.portfolio.optimizer import PortfolioAllocation, PositionAllocation

    # Reconstruct allocation
    alloc = req.allocation
    positions = [
        PositionAllocation(
            symbol=p["symbol"],
            strategy_name=p["strategy_name"],
            direction=p["direction"],
            weight=p["weight"],
            dollar_allocation=p["dollar_allocation"],
            shares=p.get("shares"),
            entry_price=p.get("entry_price"),
            stop_price=p.get("stop_price"),
            target_price=p.get("target_price"),
            risk_reward_ratio=p.get("risk_reward", 0),
            score=p.get("score", 50),
            confidence=p.get("confidence", 0.5),
        )
        for p in alloc.get("positions", [])
    ]

    allocation = PortfolioAllocation(
        as_of=date.fromisoformat(alloc["as_of"]) if "as_of" in alloc else date.today(),
        total_allocation=alloc.get("total_allocation", 0),
        cash_reserve=alloc.get("cash_reserve", 1),
        positions=positions,
        n_positions=len(positions),
    )

    constructor = TradeConstructor(use_limit_orders=req.use_limit_orders)
    plan = constructor.construct(allocation)

    return {
        "as_of": plan.as_of.isoformat(),
        "n_positions": plan.n_positions,
        "total_risk_pct": plan.total_risk_pct,
        "total_allocation_pct": plan.total_allocation_pct,
        "entry_orders": [
            {
                "symbol": o.symbol,
                "side": o.side.value,
                "type": o.order_type.value,
                "quantity": o.quantity,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "strategy": o.strategy_name,
            }
            for o in plan.orders
        ],
        "stop_orders": [
            {
                "symbol": o.symbol,
                "side": o.side.value,
                "type": o.order_type.value,
                "quantity": o.quantity,
                "stop_price": o.stop_price,
            }
            for o in plan.stop_orders
        ],
        "target_orders": [
            {
                "symbol": o.symbol,
                "side": o.side.value,
                "type": o.order_type.value,
                "quantity": o.quantity,
                "limit_price": o.limit_price,
            }
            for o in plan.target_orders
        ],
        "rejected": plan.rejected,
        "notes": plan.notes,
    }
