"""
Audit Trail — Immutable record of every trading decision and action.

The audit trail records:
1. Every order created (by which strategy, from which signal)
2. Every risk check (approved or rejected, with reasons)
3. Every order submitted to broker
4. Every fill (price, quantity, timestamp)
5. Every position opened and closed
6. Every risk alert and action taken
7. Every kill switch activation/deactivation
8. Portfolio state snapshots

This is the IMMUTABLE DECISION LOG. If it wasn't logged, it didn't happen.
The audit trail is the platform's memory of what it did and why.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Any
from enum import Enum
import json
import hashlib


class AuditActionType(Enum):
    ORDER_CREATED = "order_created"
    RISK_CHECK = "risk_check"
    RISK_APPROVED = "risk_approved"
    RISK_REJECTED = "risk_rejected"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_ADJUSTED = "stop_adjusted"
    ALERT_GENERATED = "alert_generated"
    KILL_SWITCH = "kill_switch"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    DAILY_RESET = "daily_reset"
    DRAWDOWN_EVENT = "drawdown_event"


@dataclass
class AuditEntry:
    """A single immutable audit entry."""
    id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action_type: AuditActionType = AuditActionType.ORDER_CREATED
    entity_id: str = ""  # order_id, position_id, etc.
    symbol: str = ""
    strategy_name: str = ""
    details: dict = field(default_factory=dict)
    hash: str = ""  # Hash of entry for tamper detection

    def compute_hash(self, prev_hash: str = "") -> str:
        """Compute a hash that chains entries for tamper detection."""
        content = json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "action": self.action_type.value,
            "entity_id": self.entity_id,
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "details": self.details,
            "prev_hash": prev_hash,
        }, sort_keys=True)
        self.hash = hashlib.sha256(content.encode()).hexdigest()
        return self.hash

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action_type.value,
            "entity_id": self.entity_id,
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "details": self.details,
            "hash": self.hash,
        }


class AuditTrail:
    """
    Immutable audit trail for all trading decisions and actions.

    Every entry is chained with a hash for tamper detection.
    The trail is append-only — entries can never be deleted or modified.

    Usage:
        audit = AuditTrail()
        audit.log_order_created(order_id="123", symbol="AAPL", strategy="trend_following", details={...})
        audit.log_risk_approved(order_id="123", risk_score=25, checks=[...])
        audit.log_order_filled(order_id="123", fill_price=150.25, fill_qty=100)
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []
        self._last_hash = ""
        self._counter = 0

    def _add_entry(self, entry: AuditEntry) -> AuditEntry:
        """Add an entry to the immutable trail."""
        self._counter += 1
        entry.id = f"audit_{self._counter:06d}_{int(entry.timestamp.timestamp())}"
        entry.compute_hash(self._last_hash)
        self._last_hash = entry.hash
        self._entries.append(entry)
        return entry

    # ── Order Lifecycle Logging ─────────────────────────────────────────────

    def log_order_created(
        self, order_id: str, symbol: str, strategy_name: str, details: dict
    ) -> AuditEntry:
        """Log order creation."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ORDER_CREATED,
            entity_id=order_id, symbol=symbol,
            strategy_name=strategy_name, details=details,
        ))

    def log_risk_check(
        self, order_id: str, symbol: str, approved: bool,
        risk_score: float, reasons: list[str], checks: list[dict],
    ) -> AuditEntry:
        """Log risk engine decision."""
        action = AuditActionType.RISK_APPROVED if approved else AuditActionType.RISK_REJECTED
        return self._add_entry(AuditEntry(
            action_type=action, entity_id=order_id, symbol=symbol,
            details={
                "approved": approved, "risk_score": risk_score,
                "reasons": reasons, "checks": checks,
            },
        ))

    def log_order_submitted(
        self, order_id: str, symbol: str, broker_order_id: str, details: dict
    ) -> AuditEntry:
        """Log order submission to broker."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ORDER_SUBMITTED,
            entity_id=order_id, symbol=symbol,
            details={"broker_order_id": broker_order_id, **details},
        ))

    def log_order_filled(
        self, order_id: str, symbol: str, fill_price: float,
        fill_quantity: float, fill_time: datetime,
    ) -> AuditEntry:
        """Log order fill."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ORDER_FILLED,
            entity_id=order_id, symbol=symbol,
            details={
                "fill_price": fill_price,
                "fill_quantity": fill_quantity,
                "fill_time": fill_time.isoformat(),
            },
        ))

    def log_order_cancelled(
        self, order_id: str, symbol: str, reason: str
    ) -> AuditEntry:
        """Log order cancellation."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ORDER_CANCELLED,
            entity_id=order_id, symbol=symbol,
            details={"reason": reason},
        ))

    def log_order_rejected(
        self, order_id: str, symbol: str, reason: str
    ) -> AuditEntry:
        """Log order rejection by broker."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ORDER_REJECTED,
            entity_id=order_id, symbol=symbol,
            details={"reason": reason},
        ))

    # ── Position Logging ──────────────────────────────────────────────────

    def log_position_opened(
        self, symbol: str, strategy_name: str, shares: float,
        entry_price: float, stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
    ) -> AuditEntry:
        """Log position opened."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.POSITION_OPENED,
            symbol=symbol, strategy_name=strategy_name,
            details={
                "shares": shares, "entry_price": entry_price,
                "stop_price": stop_price, "target_price": target_price,
            },
        ))

    def log_position_closed(
        self, symbol: str, strategy_name: str, shares: float,
        entry_price: float, exit_price: float, pnl: float, pnl_pct: float,
        exit_reason: str, hold_days: int,
    ) -> AuditEntry:
        """Log position closed with P&L."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.POSITION_CLOSED,
            symbol=symbol, strategy_name=strategy_name,
            details={
                "shares": shares, "entry_price": entry_price,
                "exit_price": exit_price, "pnl": pnl, "pnl_pct": pnl_pct,
                "exit_reason": exit_reason, "hold_days": hold_days,
            },
        ))

    def log_stop_adjusted(
        self, symbol: str, old_stop: float, new_stop: float, reason: str
    ) -> AuditEntry:
        """Log stop loss adjustment."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.STOP_ADJUSTED,
            symbol=symbol,
            details={"old_stop": old_stop, "new_stop": new_stop, "reason": reason},
        ))

    # ── Alert & Risk Logging ────────────────────────────────────────────────

    def log_alert(
        self, alert_type: str, symbol: str, action: str,
        message: str, severity: str = "info",
    ) -> AuditEntry:
        """Log a position monitor alert."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.ALERT_GENERATED,
            symbol=symbol,
            details={
                "alert_type": alert_type, "action": action,
                "message": message, "severity": severity,
            },
        ))

    def log_kill_switch(
        self, action: str, reason: str, triggered_by: str = "system"
    ) -> AuditEntry:
        """Log kill switch activation/deactivation."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.KILL_SWITCH,
            details={
                "action": action, "reason": reason,
                "triggered_by": triggered_by,
            },
        ))

    def log_portfolio_snapshot(
        self, equity: float, cash: float, invested: float,
        n_positions: int, daily_pnl: float, drawdown_pct: float,
    ) -> AuditEntry:
        """Log a portfolio state snapshot."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.PORTFOLIO_SNAPSHOT,
            details={
                "equity": equity, "cash": cash, "invested": invested,
                "n_positions": n_positions, "daily_pnl": daily_pnl,
                "drawdown_pct": drawdown_pct,
            },
        ))

    def log_drawdown_event(
        self, drawdown_pct: float, peak_equity: float, current_equity: float,
    ) -> AuditEntry:
        """Log a significant drawdown event."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.DRAWDOWN_EVENT,
            details={
                "drawdown_pct": drawdown_pct,
                "peak_equity": peak_equity,
                "current_equity": current_equity,
            },
        ))

    def log_daily_reset(self, reset_date: date) -> AuditEntry:
        """Log daily risk reset."""
        return self._add_entry(AuditEntry(
            action_type=AuditActionType.DAILY_RESET,
            details={"reset_date": reset_date.isoformat()},
        ))

    # ── Query Methods ──────────────────────────────────────────────────────

    def get_all_entries(self) -> list[AuditEntry]:
        """Get all audit entries (chronological)."""
        return self._entries.copy()

    def get_entries_for_symbol(self, symbol: str) -> list[AuditEntry]:
        """Get all entries for a specific symbol."""
        return [e for e in self._entries if e.symbol == symbol]

    def get_entries_for_order(self, order_id: str) -> list[AuditEntry]:
        """Get all entries for a specific order."""
        return [e for e in self._entries if e.entity_id == order_id]

    def get_entries_by_type(self, action_type: AuditActionType) -> list[AuditEntry]:
        """Get all entries of a specific type."""
        return [e for e in self._entries if e.action_type == action_type]

    def count(self) -> int:
        """Total number of entries."""
        return len(self._entries)

    def verify_integrity(self) -> bool:
        """
        Verify the hash chain is intact (tamper detection).

        Returns True if every entry's hash matches its recomputed hash.
        """
        prev_hash = ""
        for entry in self._entries:
            expected_hash = hashlib.sha256(json.dumps({
                "timestamp": entry.timestamp.isoformat(),
                "action": entry.action_type.value,
                "entity_id": entry.entity_id,
                "symbol": entry.symbol,
                "strategy": entry.strategy_name,
                "details": entry.details,
                "prev_hash": prev_hash,
            }, sort_keys=True).encode()).hexdigest()

            if entry.hash != expected_hash:
                return False
            prev_hash = entry.hash

        return True

    def export(self) -> list[dict]:
        """Export all entries as dicts (for persistence)."""
        return [e.to_dict() for e in self._entries]

    def summary(self) -> dict:
        """Get a summary of the audit trail."""
        from collections import Counter
        action_counts = Counter(e.action_type.value for e in self._entries)
        symbols = set(e.symbol for e in self._entries if e.symbol)

        return {
            "total_entries": len(self._entries),
            "actions": dict(action_counts),
            "symbols_tracked": len(symbols),
            "first_entry": self._entries[0].timestamp.isoformat() if self._entries else None,
            "last_entry": self._entries[-1].timestamp.isoformat() if self._entries else None,
            "integrity_verified": self.verify_integrity(),
        }
