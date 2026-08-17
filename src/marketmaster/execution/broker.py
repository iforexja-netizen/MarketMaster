"""
Alpaca Paper Trading Broker — Simulated order execution.

The broker integration handles:
1. Order submission (market, limit, stop, stop_limit)
2. Order status tracking (pending → filled → cancelled → rejected)
3. Position retrieval (open positions with unrealized P&L)
4. Account state (equity, cash, buying power)
5. Order cancellation

PAPER TRADING ONLY by default. The live trading flag must be explicitly
enabled in settings, and even then the Risk Engine has final authority.

The broker uses Alpaca's paper trading API:
- Base URL: https://paper-api.alpaca.markets
- Auth: API key + secret in headers
- No real money is at risk in paper mode
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Any
from enum import Enum
import asyncio
import aiohttp
import json


class OrderStatus(Enum):
    PENDING = "pending"          # Submitted, awaiting fill
    FILLED = "filled"            # Fully filled
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"        # Rejected by broker
    EXPIRED = "expired"          # Time-in-force expired


class BrokerOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"


class BrokerOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class BrokerOrder:
    """An order submitted to the broker."""
    id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: BrokerOrderSide = BrokerOrderSide.BUY
    order_type: BrokerOrderType = BrokerOrderType.MARKET
    quantity: float = 0
    filled_quantity: float = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    strategy_name: str = ""
    risk_approved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "filled_price": self.filled_price,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "strategy_name": self.strategy_name,
            "risk_approved": self.risk_approved,
        }


@dataclass
class BrokerPosition:
    """A position in the brokerage account."""
    symbol: str
    quantity: float  # positive = long, negative = short
    side: str  # "long" or "short"
    market_value: float = 0.0
    cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    current_price: float = 0.0
    entry_price: float = 0.0
    strategy_name: str = ""
    stop_price: Optional[float] = None
    target_price: Optional[float] = None


@dataclass
class AccountState:
    """Brokerage account state."""
    account_id: str = ""
    equity: float = 100_000
    cash: float = 100_000
    buying_power: float = 100_000
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    last_equity: float = 100_000
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    last_update: Optional[datetime] = None


class AlpacaPaperBroker:
    """
    Alpaca paper trading broker integration.

    PAPER TRADING ONLY. Connects to Alpaca's paper trading API.
    No real money is at risk.

    Usage:
        broker = AlpacaPaperBroker(api_key="...", api_secret="...")
        await broker.connect()
        order = await broker.submit_order(
            symbol="AAPL",
            side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET,
            quantity=10,
        )
        positions = await broker.get_positions()
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://paper-api.alpaca.markets",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._offline = True
        self._orders: dict[str, BrokerOrder] = {}  # order_id → BrokerOrder
        self._positions: dict[str, BrokerPosition] = {}  # symbol → BrokerPosition
        self._account = AccountState()

    # ── Connection Management ───────────────────────────────────────────────

    async def connect(self):
        """Initialize HTTP session and verify credentials."""
        # If no API keys, operate in offline/simulated mode
        if not self.api_key or not self.api_secret:
            self._connected = True
            self._offline = True
            return

        self._offline = False
        self._session = aiohttp.ClientSession(
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Content-Type": "application/json",
            }
        )
        self._connected = True
        # Verify connection by getting account
        try:
            await self.refresh_account()
        except Exception:
            pass  # Connection still marked as connected for paper mode

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account ─────────────────────────────────────────────────────────────

    async def refresh_account(self) -> AccountState:
        """Fetch current account state from Alpaca."""
        if not self._session:
            return self._account

        try:
            async with self._session.get(f"{self.base_url}/v2/account") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._account = AccountState(
                        account_id=data.get("id", ""),
                        equity=float(data.get("equity", 100_000)),
                        cash=float(data.get("cash", 100_000)),
                        buying_power=float(data.get("buying_power", 100_000)),
                        day_trade_count=int(data.get("daytrade_count", 0)),
                        pattern_day_trader=data.get("pattern_day_trader", False),
                        trading_blocked=data.get("trading_blocked", False),
                        last_equity=float(data.get("last_equity", 100_000)),
                        daily_pnl=float(data.get("equity", 0)) - float(data.get("last_equity", 0)),
                        daily_pnl_pct=(
                            (float(data.get("equity", 0)) / float(data.get("last_equity", 1)) - 1) * 100
                        ),
                        last_update=datetime.now(timezone.utc),
                    )
        except Exception:
            pass  # Return cached state on error

        return self._account

    @property
    def account(self) -> AccountState:
        return self._account

    # ── Order Management ────────────────────────────────────────────────────

    async def submit_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        order_type: BrokerOrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        strategy_name: str = "",
        risk_approved: bool = False,
    ) -> BrokerOrder:
        """
        Submit an order to Alpaca paper trading.

        Args:
            symbol: Ticker symbol
            side: Buy, sell, sell_short, buy_to_cover
            order_type: Market, limit, stop, stop_limit
            quantity: Number of shares
            limit_price: Required for limit and stop_limit orders
            stop_price: Required for stop and stop_limit orders
            time_in_force: day, gtc, ioc, fok
            client_order_id: Optional client-side order ID
            strategy_name: Strategy that generated this order
            risk_approved: Whether the risk engine approved this order

        Returns:
            BrokerOrder with broker-assigned ID and status
        """
        order = BrokerOrder(
            client_order_id=client_order_id or f"mm_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force.upper(),
            submitted_at=datetime.now(timezone.utc),
            strategy_name=strategy_name,
            risk_approved=risk_approved,
        )

        if self._offline or not self._session:
            # Offline/simulated mode — simulate fill
            order.status = OrderStatus.FILLED
            order.filled_price = limit_price or 100.0  # Simulated fill
            order.filled_quantity = quantity
            order.filled_at = datetime.now(timezone.utc)
            order.id = f"sim_{len(self._orders)}"
            self._orders[order.id] = order
            self._update_simulated_position(order)
            return order

        # Build Alpaca API payload
        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side.value,
            "type": order_type.value,
            "time_in_force": time_in_force,
        }
        if limit_price:
            payload["limit_price"] = str(limit_price)
        if stop_price:
            payload["stop_price"] = str(stop_price)
        if client_order_id:
            payload["client_order_id"] = client_order_id

        try:
            async with self._session.post(
                f"{self.base_url}/v2/orders", json=payload
            ) as resp:
                if resp.status == 201:
                    data = await resp.json()
                    order.id = data.get("id", "")
                    order.status = OrderStatus(data.get("status", "pending"))
                    order.filled_price = float(data.get("filled_avg_price", 0)) or None
                    order.filled_quantity = float(data.get("filled_qty", 0))
                    self._orders[order.id] = order
                else:
                    order.status = OrderStatus.REJECTED
                    error = await resp.text()
                    order.id = f"rejected_{len(self._orders)}"
                    self._orders[order.id] = order
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.id = f"error_{len(self._orders)}"
            self._orders[order.id] = order

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self._offline or not self._session:
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
                return True
            return False

        try:
            async with self._session.delete(
                f"{self.base_url}/v2/orders/{order_id}"
            ) as resp:
                if resp.status in (200, 204):
                    if order_id in self._orders:
                        self._orders[order_id].status = OrderStatus.CANCELLED
                    return True
        except Exception:
            pass
        return False

    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order status by ID."""
        if order_id in self._orders:
            return self._orders[order_id]

        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self.base_url}/v2/orders/{order_id}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    order = self._parse_alpaca_order(data)
                    self._orders[order_id] = order
                    return order
        except Exception:
            pass
        return None

    async def get_open_orders(self) -> list[BrokerOrder]:
        """Get all open (pending) orders."""
        if not self._session:
            return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

        try:
            async with self._session.get(
                f"{self.base_url}/v2/orders?status=open"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [self._parse_alpaca_order(o) for o in data]
        except Exception:
            pass
        return []

    async def get_all_orders(self) -> list[BrokerOrder]:
        """Get all orders (filled and pending)."""
        return list(self._orders.values())

    # ── Position Management ─────────────────────────────────────────────────

    async def get_positions(self) -> list[BrokerPosition]:
        """Get all open positions from Alpaca."""
        if self._offline or not self._session:
            return list(self._positions.values())

        try:
            async with self._session.get(
                f"{self.base_url}/v2/positions"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._positions.clear()
                    for pos in data:
                        p = BrokerPosition(
                            symbol=pos.get("symbol", ""),
                            quantity=float(pos.get("qty", 0)),
                            side=pos.get("side", "long"),
                            market_value=float(pos.get("market_value", 0)),
                            cost_basis=float(pos.get("cost_basis", 0)),
                            unrealized_pnl=float(pos.get("unrealized_pl", 0)),
                            unrealized_pnl_pct=float(pos.get("unrealized_plpc", 0)) * 100,
                            current_price=float(pos.get("current_price", 0)),
                            entry_price=float(pos.get("avg_entry_price", 0)),
                        )
                        self._positions[p.symbol] = p
        except Exception:
            pass

        return list(self._positions.values())

    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol."""
        if symbol in self._positions:
            return self._positions[symbol]

        if not self._session:
            return None

        try:
            async with self._session.get(
                f"{self.base_url}/v2/positions/{symbol}"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return BrokerPosition(
                        symbol=data.get("symbol", ""),
                        quantity=float(data.get("qty", 0)),
                        side=data.get("side", "long"),
                        market_value=float(data.get("market_value", 0)),
                        cost_basis=float(data.get("cost_basis", 0)),
                        unrealized_pnl=float(data.get("unrealized_pl", 0)),
                        unrealized_pnl_pct=float(data.get("unrealized_plpc", 0)) * 100,
                        current_price=float(data.get("current_price", 0)),
                        entry_price=float(data.get("avg_entry_price", 0)),
                    )
        except Exception:
            pass
        return None

    async def close_position(self, symbol: str) -> bool:
        """Close an entire position."""
        if self._offline or not self._session:
            if symbol in self._positions:
                del self._positions[symbol]
                return True
            return False

        try:
            async with self._session.delete(
                f"{self.base_url}/v2/positions/{symbol}"
            ) as resp:
                if resp.status in (200, 204):
                    if symbol in self._positions:
                        del self._positions[symbol]
                    return True
        except Exception:
            pass
        return False

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _parse_alpaca_order(self, data: dict) -> BrokerOrder:
        """Parse an Alpaca order response into BrokerOrder."""
        return BrokerOrder(
            id=data.get("id", ""),
            client_order_id=data.get("client_order_id", ""),
            symbol=data.get("symbol", ""),
            side=BrokerOrderSide(data.get("side", "buy")),
            order_type=BrokerOrderType(data.get("type", "market")),
            quantity=float(data.get("qty", 0)),
            filled_quantity=float(data.get("filled_qty", 0)),
            limit_price=float(data.get("limit_price", 0)) or None,
            stop_price=float(data.get("stop_price", 0)) or None,
            time_in_force=data.get("time_in_force", "day").upper(),
            status=OrderStatus(data.get("status", "pending")),
            filled_price=float(data.get("filled_avg_price", 0)) or None,
            filled_at=datetime.fromisoformat(data["filled_at"]) if data.get("filled_at") else None,
            submitted_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )

    def _update_simulated_position(self, order: BrokerOrder):
        """Update simulated position after a fill (offline mode)."""
        if order.status != OrderStatus.FILLED:
            return

        if order.symbol not in self._positions:
            self._positions[order.symbol] = BrokerPosition(
                symbol=order.symbol,
                quantity=0,
                side="long" if order.side == BrokerOrderSide.BUY else "short",
                entry_price=order.filled_price or 0,
                current_price=order.filled_price or 0,
                strategy_name=order.strategy_name,
            )

        pos = self._positions[order.symbol]

        if order.side in (BrokerOrderSide.BUY, BrokerOrderSide.BUY_TO_COVER):
            pos.quantity += order.filled_quantity
        else:
            pos.quantity -= order.filled_quantity

        if pos.quantity == 0:
            del self._positions[order.symbol]
