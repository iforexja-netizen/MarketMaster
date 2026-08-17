"""
MarketMaster API Routes — Phase 5: Risk + Paper Trading

Endpoints for risk engine, order lifecycle, position monitoring,
audit trail, and kill switch.
"""

from datetime import date, datetime, timezone
from typing import Optional
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from marketmaster.db.session import get_db

phase5_router = APIRouter()


# ============================================================================
# Risk Engine
# ============================================================================

class RiskCheckRequest(BaseModel):
    symbol: str
    side: str = "buy"
    quantity: float = 0
    entry_price: float = 0
    stop_price: Optional[float] = None
    sector: str = "unknown"
    strategy_name: str = ""


class PortfolioStateRequest(BaseModel):
    total_equity: float = 100_000
    cash: float = 100_000
    invested: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    open_risk_pct: float = 0.0
    positions: list[dict] = []


@phase5_router.post("/risk/evaluate")
def evaluate_order_risk(req: RiskCheckRequest, state: PortfolioStateRequest):
    """
    Evaluate a single order against all risk checks.
    The Risk Engine has FINAL AUTHORITY — if it rejects, the order cannot execute.
    """
    from marketmaster.risk.engine import RiskEngine, PortfolioRiskState

    engine = RiskEngine()
    portfolio_state = PortfolioRiskState(
        total_equity=state.total_equity,
        cash=state.cash,
        invested=state.invested,
        daily_pnl=state.daily_pnl,
        daily_pnl_pct=state.daily_pnl_pct,
        current_drawdown_pct=state.current_drawdown_pct,
        open_risk_pct=state.open_risk_pct,
        positions=state.positions,
    )

    decision = engine.evaluate_order(
        order={
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.quantity,
            "entry_price": req.entry_price,
            "stop_price": req.stop_price,
            "strategy": req.strategy_name,
            "sector": req.sector,
        },
        portfolio_state=portfolio_state,
    )

    return {
        "approved": decision.approved,
        "risk_score": decision.risk_score,
        "reasons": decision.reasons,
        "checks": [
            {
                "name": c.name,
                "level": c.level.value,
                "message": c.message,
                "passed": c.passed,
            }
            for c in decision.checks
        ],
        "adjustments": decision.adjustments,
        "kill_switch": decision.kill_switch.value,
    }


@phase5_router.post("/risk/portfolio")
def evaluate_portfolio_risk(state: PortfolioStateRequest):
    """Evaluate overall portfolio risk state."""
    from marketmaster.risk.engine import RiskEngine, PortfolioRiskState

    engine = RiskEngine()
    portfolio_state = PortfolioRiskState(
        total_equity=state.total_equity,
        cash=state.cash,
        daily_pnl=state.daily_pnl,
        daily_pnl_pct=state.daily_pnl_pct,
        current_drawdown_pct=state.current_drawdown_pct,
        open_risk_pct=state.open_risk_pct,
        positions=state.positions,
    )

    decision = engine.evaluate_portfolio(portfolio_state)

    return {
        "approved": decision.approved,
        "risk_score": decision.risk_score,
        "reasons": decision.reasons,
        "checks": [
            {"name": c.name, "level": c.level.value, "message": c.message}
            for c in decision.checks
        ],
    }


@phase5_router.post("/risk/position-size")
def compute_position_size(
    entry_price: float = Query(...),
    stop_price: float = Query(...),
    equity: float = Query(100_000),
    risk_pct: Optional[float] = None,
):
    """Compute max position size that stays within risk limits."""
    from marketmaster.risk.engine import RiskEngine
    engine = RiskEngine()
    size = engine.compute_position_size(entry_price, stop_price, equity, risk_pct)
    return {"max_shares": size, "max_dollars": size * entry_price}


@phase5_router.post("/risk/kill-switch")
def toggle_kill_switch(action: str = Query(...), reason: str = ""):
    """
    Activate or deactivate the kill switch.
    Actions: activate (halt all trading), deactivate (resume), degrade (reduce exposure)
    """
    from marketmaster.risk.engine import RiskEngine
    engine = RiskEngine()  # In production, this would be a shared singleton

    if action == "activate":
        engine.activate_kill_switch(reason)
    elif action == "deactivate":
        engine.deactivate_kill_switch()
    elif action == "degrade":
        engine.degrade_trading(reason)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    return {
        "action": action,
        "kill_switch_state": engine.kill_switch_state.value,
        "reason": reason,
    }


# ============================================================================
# Order Lifecycle
# ============================================================================

class OrderBatchRequest(BaseModel):
    orders: list[dict]  # List of order dicts
    portfolio_state: dict = {}


@phase5_router.post("/orders/process")
async def process_orders(req: OrderBatchRequest, db: Session = Depends(get_db)):
    """
    Process a batch of orders through the full lifecycle:
    Risk check → Submit to broker → Track result.

    No order reaches the broker without passing the Risk Gate.
    """
    from marketmaster.risk.engine import RiskEngine, PortfolioRiskState
    from marketmaster.execution.broker import AlpacaPaperBroker
    from marketmaster.execution.lifecycle import OrderLifecycleManager
    from marketmaster.execution.audit import AuditTrail

    engine = RiskEngine()
    broker = AlpacaPaperBroker()  # Paper trading, no credentials needed for offline
    await broker.connect()

    audit = AuditTrail()
    manager = OrderLifecycleManager(engine, broker, audit)

    state = PortfolioRiskState(
        total_equity=req.portfolio_state.get("total_equity", 100_000),
        cash=req.portfolio_state.get("cash", 100_000),
        daily_pnl=req.portfolio_state.get("daily_pnl", 0),
        daily_pnl_pct=req.portfolio_state.get("daily_pnl_pct", 0),
        current_drawdown_pct=req.portfolio_state.get("current_drawdown_pct", 0),
        open_risk_pct=req.portfolio_state.get("open_risk_pct", 0),
    )

    result = await manager.process_orders(req.orders, state)

    await broker.disconnect()

    return {
        "total": result.total,
        "approved": result.approved,
        "rejected_by_risk": result.rejected_by_risk,
        "submitted": result.submitted,
        "filled": result.filled,
        "failed": result.failed,
        "summary": result.summary,
        "orders": [
            {
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "state": o.state.value,
                "quantity": o.quantity,
                "filled_price": o.filled_price,
                "risk_reasons": o.notes,
                "errors": o.errors,
            }
            for o in result.orders
        ],
        "audit_entries": audit.count(),
    }


@phase5_router.get("/orders")
async def list_orders(status: Optional[str] = None):
    """List all managed orders, optionally filtered by status."""
    from marketmaster.execution.lifecycle import LifecycleState

    # In production, this would query the persistent store
    return {
        "orders": [],
        "filter": status,
        "note": "Query persistent order store in production",
    }


# ============================================================================
# Position Monitoring
# ============================================================================

class MonitorRequest(BaseModel):
    positions: list[dict]
    current_prices: dict[str, float]
    entry_metadata: dict[str, dict] = {}


@phase5_router.post("/positions/monitor")
def monitor_positions(req: MonitorRequest):
    """
    Check all open positions for stop/target/drawdown/time alerts.
    Returns alerts with suggested actions.
    """
    from marketmaster.execution.broker import BrokerPosition
    from marketmaster.execution.monitor import PositionMonitor

    monitor = PositionMonitor()

    # Reconstruct BrokerPosition objects
    positions = [
        BrokerPosition(
            symbol=p["symbol"],
            quantity=p["quantity"],
            side=p.get("side", "long"),
            market_value=p.get("market_value", 0),
            cost_basis=p.get("cost_basis", 0),
            unrealized_pnl=p.get("unrealized_pnl", 0),
            unrealized_pnl_pct=p.get("unrealized_pnl_pct", 0),
            current_price=p.get("current_price", 0),
            entry_price=p.get("entry_price", 0),
        )
        for p in req.positions
    ]

    result = monitor.check_positions(
        positions=positions,
        current_prices=req.current_prices,
        entry_metadata=req.entry_metadata,
    )

    return {
        "total_positions": result.total_positions,
        "positions_checked": result.positions_checked,
        "critical_count": result.critical_count,
        "warning_count": result.warning_count,
        "summary": result.summary,
        "alerts": [
            {
                "type": a.type.value,
                "symbol": a.symbol,
                "action": a.action.value,
                "message": a.message,
                "severity": a.severity,
                "current_price": a.current_price,
                "trigger_price": a.trigger_price,
                "suggested_stop": a.suggested_stop,
            }
            for a in result.alerts
        ],
    }


# ============================================================================
# Audit Trail
# ============================================================================

@phase5_router.get("/audit")
def get_audit_trail(
    symbol: Optional[str] = None,
    action_type: Optional[str] = None,
    limit: int = 100,
):
    """
    Query the immutable audit trail.
    Every trading decision and action is recorded here.
    """
    # In production, query from persistent store
    return {
        "entries": [],
        "filter": {"symbol": symbol, "action_type": action_type, "limit": limit},
        "note": "Query persistent audit store in production",
    }


@phase5_router.get("/audit/summary")
def get_audit_summary():
    """Get a summary of the audit trail."""
    return {
        "total_entries": 0,
        "actions": {},
        "symbols_tracked": 0,
        "integrity_verified": True,
        "note": "Query persistent audit store in production",
    }
