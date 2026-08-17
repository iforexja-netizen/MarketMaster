"""
Order Lifecycle Manager — Manages the full order workflow.

The lifecycle manager coordinates between:
- Trade Construction (generates orders)
- Risk Engine (evaluates and approves/rejects orders)
- Broker (submits and tracks orders)
- Audit Log (records every action)

Order states:
  CREATED → RISK_CHECKED → SUBMITTED → FILLED / REJECTED / CANCELLED / EXPIRED

Every transition is logged to the immutable decision log.
No order reaches the broker without passing the Risk Gate.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Any
from enum import Enum
import asyncio

from marketmaster.execution.broker import (
    AlpacaPaperBroker, BrokerOrder, BrokerOrderSide, BrokerOrderType,
    OrderStatus, BrokerPosition, AccountState,
)
from marketmaster.risk.engine import RiskEngine, RiskDecision, PortfolioRiskState


class LifecycleState(Enum):
    CREATED = "created"           # Order created by trade construction
    RISK_CHECKED = "risk_checked"  # Passed through risk engine
    RISK_REJECTED = "risk_rejected" # Rejected by risk engine
    SUBMITTED = "submitted"        # Submitted to broker
    FILLED = "filled"              # Filled by broker
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"          # Rejected by broker
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class ManagedOrder:
    """An order tracked through its full lifecycle."""
    # Identity
    id: str = ""
    symbol: str = ""
    side: str = "buy"  # buy, sell, sell_short, buy_to_cover
    order_type: str = "market"  # market, limit, stop, stop_limit
    quantity: float = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_name: str = ""

    # Lifecycle
    state: LifecycleState = LifecycleState.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risk_checked_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Risk
    risk_decision: Optional[RiskDecision] = None
    risk_adjusted_quantity: Optional[float] = None

    # Broker
    broker_order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0

    # Audit
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class LifecycleResult:
    """Result of processing a batch of orders."""
    total: int = 0
    approved: int = 0
    rejected_by_risk: int = 0
    submitted: int = 0
    filled: int = 0
    failed: int = 0
    orders: list[ManagedOrder] = field(default_factory=list)
    summary: str = ""


class OrderLifecycleManager:
    """
    Manages the full order lifecycle from creation to fill.

    Usage:
        manager = OrderLifecycleManager(risk_engine, broker)
        result = await manager.process_orders(orders, portfolio_state)
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        broker: AlpacaPaperBroker,
        audit_log: Optional[Any] = None,
    ):
        self.risk_engine = risk_engine
        self.broker = broker
        self.audit_log = audit_log
        self._managed_orders: dict[str, ManagedOrder] = {}

    async def process_orders(
        self,
        orders: list[dict],  # List of order dicts from trade construction
        portfolio_state: PortfolioRiskState,
        as_of: Optional[datetime] = None,
    ) -> LifecycleResult:
        """
        Process a batch of orders through the full lifecycle:
        1. Create managed orders
        2. Run each through the Risk Engine
        3. Submit approved orders to the broker
        4. Track results

        Returns LifecycleResult with counts and all managed orders.
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        result = LifecycleResult(total=len(orders))

        for order_data in orders:
            managed = ManagedOrder(
                id=f"mm_{order_data['symbol']}_{int(as_of.timestamp())}_{result.total}",
                symbol=order_data["symbol"],
                side=order_data.get("side", "buy"),
                order_type=order_data.get("order_type", "market"),
                quantity=order_data.get("quantity", 0),
                limit_price=order_data.get("limit_price"),
                stop_price=order_data.get("stop_price"),
                strategy_name=order_data.get("strategy_name", ""),
            )

            # ── Step 1: Risk Check ───────────────────────────────────────
            risk_order = {
                "symbol": managed.symbol,
                "side": managed.side,
                "quantity": managed.quantity,
                "entry_price": managed.limit_price or order_data.get("entry_price", 0),
                "stop_price": managed.stop_price,
                "strategy": managed.strategy_name,
                "sector": order_data.get("sector", "unknown"),
            }

            decision = self.risk_engine.evaluate_order(
                order=risk_order,
                portfolio_state=portfolio_state,
                as_of=as_of,
                data_timestamp=as_of,
            )

            managed.risk_decision = decision
            managed.risk_checked_at = datetime.now(timezone.utc)

            if not decision.approved:
                managed.state = LifecycleState.RISK_REJECTED
                managed.notes.extend(decision.reasons)
                result.rejected_by_risk += 1
                result.orders.append(managed)
                self._managed_orders[managed.id] = managed
                self._log_audit(managed, "RISK_REJECTED", decision.reasons)
                continue

            # Apply risk adjustments (position size reduction)
            if decision.adjustments.get("size_multiplier"):
                multiplier = decision.adjustments["size_multiplier"]
                managed.risk_adjusted_quantity = managed.quantity * multiplier
                managed.quantity = managed.risk_adjusted_quantity
                managed.notes.append(f"Size adjusted by risk engine: ×{multiplier:.2f}")

            result.approved += 1
            managed.state = LifecycleState.RISK_CHECKED

            # ── Step 2: Submit to Broker ─────────────────────────────────
            try:
                broker_order = await self.broker.submit_order(
                    symbol=managed.symbol,
                    side=BrokerOrderSide(managed.side),
                    order_type=BrokerOrderType(managed.order_type),
                    quantity=managed.quantity,
                    limit_price=managed.limit_price,
                    stop_price=managed.stop_price,
                    strategy_name=managed.strategy_name,
                    risk_approved=True,
                )

                managed.broker_order_id = broker_order.id
                managed.submitted_at = datetime.now(timezone.utc)

                if broker_order.status == OrderStatus.FILLED:
                    managed.state = LifecycleState.FILLED
                    managed.filled_price = broker_order.filled_price
                    managed.filled_quantity = broker_order.filled_quantity
                    managed.filled_at = broker_order.filled_at
                    result.filled += 1
                elif broker_order.status == OrderStatus.REJECTED:
                    managed.state = LifecycleState.REJECTED
                    managed.errors.append("Broker rejected order")
                    result.failed += 1
                else:
                    managed.state = LifecycleState.SUBMITTED
                    result.submitted += 1

            except Exception as e:
                managed.state = LifecycleState.ERROR
                managed.errors.append(str(e))
                result.failed += 1

            result.orders.append(managed)
            self._managed_orders[managed.id] = managed
            self._log_audit(managed, managed.state.value, [])

        # ── Summary ──────────────────────────────────────────────────────
        result.summary = (
            f"Processed {result.total} orders: "
            f"{result.approved} approved, {result.rejected_by_risk} rejected by risk, "
            f"{result.submitted} submitted, {result.filled} filled, {result.failed} failed"
        )

        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self._managed_orders:
            return False

        managed = self._managed_orders[order_id]
        if managed.state not in (LifecycleState.SUBMITTED, LifecycleState.CREATED):
            return False

        if managed.broker_order_id:
            success = await self.broker.cancel_order(managed.broker_order_id)
            if success:
                managed.state = LifecycleState.CANCELLED
                self._log_audit(managed, "CANCELLED", ["User cancelled"])
                return True
        else:
            managed.state = LifecycleState.CANCELLED
            self._log_audit(managed, "CANCELLED", ["Cancelled before submission"])
            return True

        return False

    async def sync_order_status(self) -> list[ManagedOrder]:
        """Sync all pending orders with broker status."""
        updated = []
        for managed in self._managed_orders.values():
            if managed.state == LifecycleState.SUBMITTED and managed.broker_order_id:
                broker_order = await self.broker.get_order(managed.broker_order_id)
                if broker_order:
                    if broker_order.status == OrderStatus.FILLED:
                        managed.state = LifecycleState.FILLED
                        managed.filled_price = broker_order.filled_price
                        managed.filled_quantity = broker_order.filled_quantity
                        managed.filled_at = broker_order.filled_at
                        updated.append(managed)
                    elif broker_order.status == OrderStatus.CANCELLED:
                        managed.state = LifecycleState.CANCELLED
                        updated.append(managed)
                    elif broker_order.status == OrderStatus.REJECTED:
                        managed.state = LifecycleState.REJECTED
                        updated.append(managed)

        return updated

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get a managed order by ID."""
        return self._managed_orders.get(order_id)

    def get_all_orders(self) -> list[ManagedOrder]:
        """Get all managed orders."""
        return list(self._managed_orders.values())

    def get_open_orders(self) -> list[ManagedOrder]:
        """Get all open (pending) orders."""
        return [
            o for o in self._managed_orders.values()
            if o.state in (LifecycleState.CREATED, LifecycleState.RISK_CHECKED, LifecycleState.SUBMITTED)
        ]

    def _log_audit(self, order: ManagedOrder, action: str, reasons: list[str]):
        """Log an audit entry (to immutable decision log if available)."""
        if self.audit_log:
            try:
                # Use AuditTrail's specific logging methods based on action
                if action.upper() == "RISK_REJECTED":
                    self.audit_log.log_risk_check(
                        order_id=order.id, symbol=order.symbol,
                        approved=False, risk_score=order.risk_decision.risk_score if order.risk_decision else 100,
                        reasons=reasons, checks=[],
                    )
                elif action.upper() == "RISK_CHECKED":
                    self.audit_log.log_risk_check(
                        order_id=order.id, symbol=order.symbol,
                        approved=True, risk_score=order.risk_decision.risk_score if order.risk_decision else 0,
                        reasons=reasons, checks=[],
                    )
                elif action.upper() in ("FILLED", "SUBMITTED", "CREATED"):
                    if action.upper() == "FILLED":
                        self.audit_log.log_order_filled(
                            order_id=order.id, symbol=order.symbol,
                            fill_price=order.filled_price or 0,
                            fill_quantity=order.filled_quantity,
                            fill_time=order.filled_at or datetime.now(timezone.utc),
                        )
                    elif action.upper() == "SUBMITTED" and order.broker_order_id:
                        self.audit_log.log_order_submitted(
                            order_id=order.id, symbol=order.symbol,
                            broker_order_id=order.broker_order_id,
                            details={"strategy": order.strategy_name},
                        )
                    else:
                        self.audit_log.log_order_created(
                            order_id=order.id, symbol=order.symbol,
                            strategy_name=order.strategy_name,
                            details={"side": order.side, "qty": order.quantity},
                        )
                elif action.upper() == "CANCELLED":
                    self.audit_log.log_order_cancelled(order.id, order.symbol, "Cancelled" )
                elif action.upper() == "REJECTED":
                    self.audit_log.log_order_rejected(order.id, order.symbol, "Broker rejected")
            except Exception:
                pass
