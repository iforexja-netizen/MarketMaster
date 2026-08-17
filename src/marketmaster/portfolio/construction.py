"""
Trade Construction — Convert portfolio allocations to executable orders.

The trade construction module takes a PortfolioAllocation and produces
concrete order instructions with:
- Entry type (market, limit, stop)
- Position sizing in shares/dollars
- Stop loss and take profit orders
- Risk-adjusted position verification

This is the last step before the Risk Gate. The Risk Gate has FINAL AUTHORITY
and can reject any order.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional
from enum import Enum

from marketmaster.portfolio.optimizer import PositionAllocation, PortfolioAllocation


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"


@dataclass
class Order:
    """A concrete order instruction."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC
    strategy_name: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class TradePlan:
    """Complete trade plan for a portfolio allocation."""
    as_of: date
    orders: list[Order] = field(default_factory=list)
    stop_orders: list[Order] = field(default_factory=list)
    target_orders: list[Order] = field(default_factory=list)
    total_risk_pct: float = 0.0
    total_allocation_pct: float = 0.0
    n_positions: int = 0
    rejected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class TradeConstructor:
    """
    Converts portfolio allocations to executable order instructions.

    Usage:
        constructor = TradeConstructor()
        plan = constructor.construct(allocation, as_of=date(2025, 6, 1))
    """

    def __init__(self, use_limit_orders: bool = True, limit_buffer_pct: float = 0.1):
        self.use_limit_orders = use_limit_orders
        self.limit_buffer_pct = limit_buffer_pct  # 0.1% above entry for limit buys

    def construct(
        self,
        allocation: PortfolioAllocation,
        as_of: Optional[date] = None,
    ) -> TradePlan:
        """
        Construct a complete trade plan from portfolio allocation.

        For each position, generates:
        1. Entry order (market or limit)
        2. Stop loss order (stop market)
        3. Take profit order (limit)

        Returns TradePlan with all orders and risk summary.
        """
        if as_of is None:
            as_of = date.today()

        plan = TradePlan(as_of=as_of, n_positions=allocation.n_positions)

        total_risk = 0.0

        for pos in allocation.positions:
            if pos.entry_price is None or pos.entry_price <= 0:
                plan.rejected.append(f"{pos.symbol}: no entry price")
                continue

            # ── Entry Order ───────────────────────────────────────────────
            side = OrderSide.BUY if pos.direction == "long" else OrderSide.SELL_SHORT

            if self.use_limit_orders:
                # Limit order at entry price (or slightly above for buys)
                if pos.direction == "long":
                    limit_price = pos.entry_price * (1 + self.limit_buffer_pct / 100)
                else:
                    limit_price = pos.entry_price * (1 - self.limit_buffer_pct / 100)

                entry_order = Order(
                    symbol=pos.symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=pos.shares or 0,
                    limit_price=round(limit_price, 2),
                    strategy_name=pos.strategy_name,
                    notes=[f"Limit entry for {pos.strategy_name}"],
                )
            else:
                entry_order = Order(
                    symbol=pos.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=pos.shares or 0,
                    strategy_name=pos.strategy_name,
                    notes=[f"Market entry for {pos.strategy_name}"],
                )

            plan.orders.append(entry_order)

            # ── Stop Loss Order ───────────────────────────────────────────
            if pos.stop_price and pos.stop_price > 0:
                stop_side = OrderSide.SELL if pos.direction == "long" else OrderSide.BUY_TO_COVER
                stop_order = Order(
                    symbol=pos.symbol,
                    side=stop_side,
                    order_type=OrderType.STOP,
                    quantity=pos.shares or 0,
                    stop_price=round(pos.stop_price, 2),
                    strategy_name=pos.strategy_name,
                    notes=[f"Stop loss at {pos.stop_price:.2f}"],
                )
                plan.stop_orders.append(stop_order)

                # Track risk
                if pos.entry_price and pos.shares:
                    risk_per_share = abs(pos.entry_price - pos.stop_price)
                    total_risk += risk_per_share * pos.shares

            # ── Take Profit Order ────────────────────────────────────────
            if pos.target_price and pos.target_price > 0:
                target_side = OrderSide.SELL if pos.direction == "long" else OrderSide.BUY_TO_COVER
                target_order = Order(
                    symbol=pos.symbol,
                    side=target_side,
                    order_type=OrderType.LIMIT,
                    quantity=pos.shares or 0,
                    limit_price=round(pos.target_price, 2),
                    strategy_name=pos.strategy_name,
                    notes=[f"Take profit at {pos.target_price:.2f}"],
                )
                plan.target_orders.append(target_order)

        # ── Risk Summary ──────────────────────────────────────────────────
        plan.total_risk_pct = (total_risk / 100_000) * 100 if total_risk > 0 else 0
        plan.total_allocation_pct = allocation.total_allocation * 100

        plan.notes.append(f"Entry orders: {len(plan.orders)}")
        plan.notes.append(f"Stop orders: {len(plan.stop_orders)}")
        plan.notes.append(f"Target orders: {len(plan.target_orders)}")
        plan.notes.append(f"Total portfolio risk: {plan.total_risk_pct:.2f}%")
        plan.notes.append(f"Total allocation: {plan.total_allocation_pct:.1f}%")

        return plan
