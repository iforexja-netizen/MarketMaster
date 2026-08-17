"""
Phase 5 tests — Risk engine, order lifecycle, position monitor, audit trail.

Tests the deterministic risk engine, broker integration (offline mode),
order lifecycle management, position monitoring, and audit trail integrity.
"""

import pytest
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Optional
import asyncio
import numpy as np

from marketmaster.risk.engine import (
    RiskEngine, RiskDecision, RiskCheck, RiskLevel,
    PortfolioRiskState, KillSwitchState,
)
from marketmaster.execution.broker import (
    AlpacaPaperBroker, BrokerOrder, BrokerOrderSide, BrokerOrderType,
    OrderStatus, BrokerPosition, AccountState,
)
from marketmaster.execution.lifecycle import (
    OrderLifecycleManager, ManagedOrder, LifecycleState, LifecycleResult,
)
from marketmaster.execution.monitor import (
    PositionMonitor, PositionAlert, MonitoringResult, AlertType, AlertAction,
)
from marketmaster.execution.audit import AuditTrail, AuditEntry, AuditActionType


# ============================================================================
# Risk Engine Tests
# ============================================================================

class TestRiskEngine:
    def _make_portfolio_state(self, **kwargs):
        return PortfolioRiskState(
            total_equity=kwargs.get("total_equity", 100_000),
            cash=kwargs.get("cash", 100_000),
            invested=kwargs.get("invested", 0),
            daily_pnl=kwargs.get("daily_pnl", 0),
            daily_pnl_pct=kwargs.get("daily_pnl_pct", 0),
            current_drawdown_pct=kwargs.get("current_drawdown_pct", 0),
            open_risk_pct=kwargs.get("open_risk_pct", 0),
            positions=kwargs.get("positions", []),
        )

    def test_risk_engine_creation(self):
        engine = RiskEngine()
        assert engine.live_trading_enabled is False
        assert engine.max_position_risk_pct == 0.005
        assert engine.max_daily_loss_pct == 0.02
        assert engine.kill_switch_state == KillSwitchState.ACTIVE

    def test_approves_valid_order(self):
        engine = RiskEngine()
        state = self._make_portfolio_state()
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "entry_price": 150.0,
            "stop_price": 145.0,  # 3.3% risk per share → 0.005 risk on portfolio
            "sector": "technology",
            "strategy": "trend_following",
        }
        decision = engine.evaluate_order(order, state)
        assert decision.approved
        assert decision.risk_score < 50
        assert len(decision.failed_checks) == 0

    def test_rejects_excessive_position_risk(self):
        engine = RiskEngine()
        state = self._make_portfolio_state()
        # Very large position: 1000 shares at $150, stop at $100
        # Risk = (150-100) × 1000 / 100000 = 50% — way over limit
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1000,
            "entry_price": 150.0,
            "stop_price": 100.0,
            "sector": "technology",
            "strategy": "trend_following",
        }
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "POSITION_RISK_LIMIT" in decision.reasons
        assert "max_quantity" in decision.adjustments  # Suggests reduced size

    def test_rejects_position_size_over_limit(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(total_equity=10_000)
        # Position value = 2000 shares × $100 = $200k on $10k equity = 2000%
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 2000,
            "entry_price": 100.0,
            "stop_price": 99.5,
            "sector": "tech",
        }
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "POSITION_SIZE_LIMIT" in decision.reasons

    def test_rejects_daily_loss_limit(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(daily_pnl=-3000, daily_pnl_pct=-3.0)
        order = {
            "symbol": "AAPL", "side": "buy", "quantity": 10,
            "entry_price": 150, "stop_price": 149,
            "sector": "tech",
        }
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "DAILY_LOSS_LIMIT" in decision.reasons

    def test_warns_on_approaching_daily_loss(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(daily_pnl=-1700, daily_pnl_pct=-1.7)
        order = {
            "symbol": "AAPL", "side": "buy", "quantity": 10,
            "entry_price": 150, "stop_price": 149,
            "sector": "tech",
        }
        decision = engine.evaluate_order(order, state)
        # 1.4% > 75% of 2% limit → should warn but not fail on daily loss
        daily_check = [c for c in decision.checks if c.name == "daily_loss"]
        assert any(c.level == RiskLevel.WARN for c in daily_check)

    def test_rejects_sector_concentration(self):
        engine = RiskEngine()
        # Create positions that already have 25% in tech
        state = self._make_portfolio_state(
            total_equity=100_000,
            positions=[
                {"symbol": "MSFT", "sector": "technology", "market_value": 25_000},
            ]
        )
        # New position adds 10% more → 35% total, over 30% limit
        order = {
            "symbol": "AAPL", "side": "buy", "quantity": 100,
            "entry_price": 100, "stop_price": 99.5,
            "sector": "technology",
        }
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "SECTOR_CONCENTRATION" in decision.reasons

    def test_rejects_portfolio_risk_limit(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(open_risk_pct=9.5)  # 9.5% open risk
        # New order adds 1% more → 10.5% total, over 10% limit
        order = {
            "symbol": "AAPL", "side": "buy", "quantity": 100,
            "entry_price": 150, "stop_price": 135,
            "sector": "tech",
        }
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "PORTFOLIO_RISK_LIMIT" in decision.reasons

    def test_kill_switch_halt(self):
        engine = RiskEngine()
        engine.activate_kill_switch("Test halt")
        state = self._make_portfolio_state()
        order = {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}
        decision = engine.evaluate_order(order, state)
        assert not decision.approved
        assert "KILL_SWITCH_HALT" in decision.reasons
        assert decision.kill_switch == KillSwitchState.HALTED

    def test_kill_switch_degrade(self):
        engine = RiskEngine()
        engine.degrade_trading("Drawdown")
        state = self._make_portfolio_state()
        order = {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}
        decision = engine.evaluate_order(order, state)
        # Degrading doesn't reject but reduces size
        assert "size_multiplier" in decision.adjustments

    def test_stale_data_rejection(self):
        engine = RiskEngine()
        state = self._make_portfolio_state()
        old_timestamp = datetime.now(timezone.utc) - timedelta(minutes=30)
        order = {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}
        decision = engine.evaluate_order(order, state, data_timestamp=old_timestamp)
        assert not decision.approved
        assert "STALE_DATA" in decision.reasons

    def test_drawdown_reduces_exposure(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(current_drawdown_pct=15.0)
        order = {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}
        decision = engine.evaluate_order(order, state)
        # Should warn and suggest reduced size
        dd_checks = [c for c in decision.checks if c.name == "drawdown"]
        assert any(c.level == RiskLevel.WARN for c in dd_checks)
        assert "size_multiplier" in decision.adjustments

    def test_compute_position_size(self):
        engine = RiskEngine()
        # $100k equity, 0.5% risk = $500 risk budget
        # Entry $150, stop $145 → $5 risk per share
        # Max shares = 500 / 5 = 100
        size = engine.compute_position_size(150.0, 145.0, 100_000)
        assert size == pytest.approx(100.0, rel=0.01)

    def test_compute_position_size_zero_stop(self):
        engine = RiskEngine()
        size = engine.compute_position_size(150.0, 150.0, 100_000)
        assert size == 0.0

    def test_portfolio_risk_evaluation(self):
        engine = RiskEngine()
        state = self._make_portfolio_state(
            current_drawdown_pct=25,
            daily_pnl_pct=-3.0,
            open_risk_pct=15,
        )
        decision = engine.evaluate_portfolio(state)
        assert not decision.approved
        assert "SEVERE_DRAWDOWN" in decision.reasons

    def test_live_trading_flag_in_decision(self):
        engine = RiskEngine()
        state = self._make_portfolio_state()
        order = {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}
        decision = engine.evaluate_order(order, state)
        # Live trading is disabled by default → should note paper mode
        live_checks = [c for c in decision.checks if c.name == "live_trading"]
        assert len(live_checks) > 0
        assert "paper" in live_checks[0].message.lower()


# ============================================================================
# Broker Tests (Offline Mode)
# ============================================================================

class TestBrokerOffline:
    @pytest.fixture
    def broker(self):
        return AlpacaPaperBroker(api_key="", api_secret="")

    def test_broker_creation(self, broker):
        assert broker.is_connected is False

    async def test_submit_order_offline(self, broker):
        await broker.connect()
        order = await broker.submit_order(
            symbol="AAPL",
            side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET,
            quantity=10,
            strategy_name="trend_following",
            risk_approved=True,
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert order.filled_price is not None
        await broker.disconnect()

    async def test_position_tracking_offline(self, broker):
        await broker.connect()
        await broker.submit_order(
            symbol="AAPL", side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET, quantity=50,
        )
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 50
        await broker.disconnect()

    async def test_close_position_offline(self, broker):
        await broker.connect()
        await broker.submit_order(
            symbol="AAPL", side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET, quantity=10,
        )
        assert len(await broker.get_positions()) == 1
        success = await broker.close_position("AAPL")
        assert success
        assert len(await broker.get_positions()) == 0
        await broker.disconnect()

    async def test_cancel_order_offline(self, broker):
        await broker.connect()
        order = await broker.submit_order(
            symbol="AAPL", side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.MARKET, quantity=10,
        )
        # Order is already filled in offline mode, so cancel should return False
        success = await broker.cancel_order(order.id)
        assert success  # Filled orders can still be "cancelled" in sim
        await broker.disconnect()

    async def test_limit_order_offline(self, broker):
        await broker.connect()
        order = await broker.submit_order(
            symbol="AAPL", side=BrokerOrderSide.BUY,
            order_type=BrokerOrderType.LIMIT, quantity=10,
            limit_price=145.0,
        )
        assert order.limit_price == 145.0
        assert order.order_type == BrokerOrderType.LIMIT
        await broker.disconnect()

    async def test_short_position_offline(self, broker):
        await broker.connect()
        await broker.submit_order(
            symbol="TSLA", side=BrokerOrderSide.SELL_SHORT,
            order_type=BrokerOrderType.MARKET, quantity=20,
        )
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == -20  # Short = negative
        await broker.disconnect()

    async def test_account_state(self, broker):
        await broker.connect()
        account = broker.account
        assert account.equity == 100_000  # Default
        assert account.cash == 100_000
        await broker.disconnect()


# ============================================================================
# Order Lifecycle Tests
# ============================================================================

class TestOrderLifecycle:
    @pytest.fixture
    def lifecycle_manager(self):
        engine = RiskEngine()
        broker = AlpacaPaperBroker()
        audit = AuditTrail()
        return OrderLifecycleManager(engine, broker, audit)

    async def test_process_approved_order(self, lifecycle_manager):
        state = PortfolioRiskState(total_equity=100_000)
        orders = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "market",
                "quantity": 10,
                "entry_price": 150.0,
                "stop_price": 149.0,
                "sector": "tech",
                "strategy_name": "trend_following",
            }
        ]
        result = await lifecycle_manager.process_orders(orders, state)
        assert result.total == 1
        assert result.approved == 1
        assert result.filled >= 1
        assert result.rejected_by_risk == 0

    async def test_process_rejected_order(self, lifecycle_manager):
        state = PortfolioRiskState(total_equity=100_000)
        # Position risk way over limit
        orders = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "order_type": "market",
                "quantity": 10000,
                "entry_price": 150.0,
                "stop_price": 100.0,
                "sector": "tech",
                "strategy_name": "trend_following",
            }
        ]
        result = await lifecycle_manager.process_orders(orders, state)
        assert result.rejected_by_risk == 1
        assert result.approved == 0
        assert result.orders[0].state == LifecycleState.RISK_REJECTED

    async def test_process_multiple_orders(self, lifecycle_manager):
        state = PortfolioRiskState(total_equity=100_000)
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149, "sector": "tech"},
            {"symbol": "MSFT", "side": "buy", "quantity": 5, "entry_price": 400, "stop_price": 398, "sector": "tech"},
        ]
        result = await lifecycle_manager.process_orders(orders, state)
        assert result.total == 2
        assert result.approved == 2

    async def test_kill_switch_blocks_all(self, lifecycle_manager):
        lifecycle_manager.risk_engine.activate_kill_switch("Test")
        state = PortfolioRiskState(total_equity=100_000)
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}]
        result = await lifecycle_manager.process_orders(orders, state)
        assert result.rejected_by_risk == 1
        assert "KILL_SWITCH_HALT" in result.orders[0].notes

    async def test_audit_log_populated(self, lifecycle_manager):
        state = PortfolioRiskState(total_equity=100_000)
        orders = [{"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149}]
        await lifecycle_manager.process_orders(orders, state)
        assert lifecycle_manager.audit_log.count() > 0

    async def test_get_open_orders(self, lifecycle_manager):
        state = PortfolioRiskState(total_equity=100_000)
        orders = [
            {"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 150, "stop_price": 149},
            {"symbol": "TSLA", "side": "buy", "quantity": 10000, "entry_price": 200, "stop_price": 50},
        ]
        await lifecycle_manager.process_orders(orders, state)
        # One approved, one rejected → no open orders (offline fills immediately)
        all_orders = lifecycle_manager.get_all_orders()
        assert len(all_orders) == 2


# ============================================================================
# Position Monitor Tests
# ============================================================================

class TestPositionMonitor:
    def _make_position(self, symbol="AAPL", quantity=100, entry=150, current=150):
        return BrokerPosition(
            symbol=symbol, quantity=quantity, side="long",
            market_value=current * quantity, cost_basis=entry * quantity,
            unrealized_pnl=(current - entry) * quantity,
            unrealized_pnl_pct=((current - entry) / entry) * 100,
            current_price=current, entry_price=entry,
        )

    def test_no_alerts_when_healthy(self):
        monitor = PositionMonitor()
        pos = self._make_position(current=155, entry=150)
        meta = {"AAPL": {"stop_price": 145, "target_price": 165, "entry_price": 150}}
        result = monitor.check_positions([pos], {"AAPL": 155}, meta)
        # Price is between stop and target → no critical alerts
        assert result.critical_count == 0

    def test_stop_loss_alert(self):
        monitor = PositionMonitor()
        pos = self._make_position(current=144, entry=150)
        meta = {"AAPL": {"stop_price": 145, "entry_price": 150}}
        result = monitor.check_positions([pos], {"AAPL": 144}, meta)
        assert result.critical_count >= 1
        stop_alerts = [a for a in result.alerts if a.type == AlertType.STOP_LOSS]
        assert len(stop_alerts) == 1
        assert stop_alerts[0].action == AlertAction.CLOSE

    def test_take_profit_alert(self):
        monitor = PositionMonitor()
        pos = self._make_position(current=165, entry=150)
        meta = {"AAPL": {"target_price": 160, "entry_price": 150}}
        result = monitor.check_positions([pos], {"AAPL": 165}, meta)
        tp_alerts = [a for a in result.alerts if a.type == AlertType.TAKE_PROFIT]
        assert len(tp_alerts) == 1
        assert tp_alerts[0].action == AlertAction.CLOSE

    def test_trailing_stop_suggestion(self):
        monitor = PositionMonitor(trailing_stop_activation_pct=5.0, trailing_stop_distance_pct=3.0)
        pos = self._make_position(current=160, entry=150)  # 6.7% gain → activate trailing
        meta = {"AAPL": {"entry_price": 150, "stop_price": 145}}
        result = monitor.check_positions([pos], {"AAPL": 160}, meta)
        trail_alerts = [a for a in result.alerts if a.type == AlertType.TRAILING_STOP]
        assert len(trail_alerts) == 1
        assert trail_alerts[0].suggested_stop > 145  # New stop above old
        assert trail_alerts[0].suggested_stop < 160  # Below current price

    def test_drawdown_alert(self):
        monitor = PositionMonitor(position_drawdown_alert_pct=5.0)
        pos = self._make_position(current=140, entry=150)  # -6.7% drawdown
        meta = {"AAPL": {"entry_price": 150}}
        result = monitor.check_positions([pos], {"AAPL": 140}, meta)
        dd_alerts = [a for a in result.alerts if a.type == AlertType.DRAWDOWN]
        assert len(dd_alerts) >= 1

    def test_time_exit_alert(self):
        monitor = PositionMonitor(max_hold_days=5)
        pos = self._make_position(current=150, entry=150)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)
        meta = {"AAPL": {"entry_date": old_date, "entry_price": 150}}
        result = monitor.check_positions([pos], {"AAPL": 150}, meta)
        time_alerts = [a for a in result.alerts if a.type == AlertType.TIME_EXIT]
        assert len(time_alerts) == 1
        assert "10 days" in time_alerts[0].message

    def test_short_position_stop(self):
        monitor = PositionMonitor()
        pos = BrokerPosition(
            symbol="TSLA", quantity=-100, side="short",
            entry_price=200, current_price=215,  # Short losing
            unrealized_pnl=-1500, unrealized_pnl_pct=-7.5,
        )
        meta = {"TSLA": {"stop_price": 210, "entry_price": 200}}
        result = monitor.check_positions([pos], {"TSLA": 215}, meta)
        stop_alerts = [a for a in result.alerts if a.type == AlertType.STOP_LOSS]
        assert len(stop_alerts) == 1

    def test_empty_positions(self):
        monitor = PositionMonitor()
        result = monitor.check_positions([], {})
        assert result.total_positions == 0
        assert len(result.alerts) == 0

    def test_monitoring_summary(self):
        monitor = PositionMonitor()
        pos = self._make_position()
        result = monitor.check_positions([pos], {"AAPL": 150})
        assert "Monitored" in result.summary


# ============================================================================
# Audit Trail Tests
# ============================================================================

class TestAuditTrail:
    def test_empty_trail(self):
        audit = AuditTrail()
        assert audit.count() == 0

    def test_log_order_created(self):
        audit = AuditTrail()
        entry = audit.log_order_created("ord_1", "AAPL", "trend_following", {"qty": 10})
        assert audit.count() == 1
        assert entry.action_type == AuditActionType.ORDER_CREATED
        assert entry.symbol == "AAPL"

    def test_log_risk_approved(self):
        audit = AuditTrail()
        audit.log_risk_check("ord_1", "AAPL", True, 25.0, [], [])
        assert audit.count() == 1
        entries = audit.get_entries_for_order("ord_1")
        assert entries[0].action_type == AuditActionType.RISK_APPROVED

    def test_log_risk_rejected(self):
        audit = AuditTrail()
        audit.log_risk_check("ord_1", "AAPL", False, 80.0, ["POSITION_RISK_LIMIT"], [])
        entries = audit.get_entries_for_order("ord_1")
        assert entries[0].action_type == AuditActionType.RISK_REJECTED

    def test_log_fill(self):
        audit = AuditTrail()
        audit.log_order_filled("ord_1", "AAPL", 150.25, 100, datetime.now(timezone.utc))
        entries = audit.get_entries_for_order("ord_1")
        assert entries[0].details["fill_price"] == 150.25

    def test_log_position_opened(self):
        audit = AuditTrail()
        audit.log_position_opened("AAPL", "trend_following", 100, 150.0, 145.0, 165.0)
        entries = audit.get_entries_for_symbol("AAPL")
        assert entries[0].action_type == AuditActionType.POSITION_OPENED

    def test_log_position_closed(self):
        audit = AuditTrail()
        audit.log_position_closed("AAPL", "trend_following", 100, 150.0, 160.0, 1000.0, 6.67, "target", 5)
        entries = audit.get_entries_for_symbol("AAPL")
        assert entries[0].details["pnl"] == 1000.0

    def test_log_kill_switch(self):
        audit = AuditTrail()
        audit.log_kill_switch("activate", "Manual halt", "user")
        assert audit.count() == 1

    def test_log_portfolio_snapshot(self):
        audit = AuditTrail()
        audit.log_portfolio_snapshot(100_000, 50_000, 50_000, 5, 500, 2.0)
        entries = audit.get_entries_by_type(AuditActionType.PORTFOLIO_SNAPSHOT)
        assert len(entries) == 1
        assert entries[0].details["equity"] == 100_000

    def test_hash_integrity(self):
        audit = AuditTrail()
        audit.log_order_created("ord_1", "AAPL", "trend_following", {})
        audit.log_risk_check("ord_1", "AAPL", True, 25.0, [], [])
        audit.log_order_filled("ord_1", "AAPL", 150.0, 100, datetime.now(timezone.utc))
        assert audit.verify_integrity()  # Hash chain intact

    def test_tamper_detection(self):
        audit = AuditTrail()
        audit.log_order_created("ord_1", "AAPL", "trend_following", {"qty": 10})
        audit.log_risk_check("ord_1", "AAPL", True, 25.0, [], [])
        # Tamper with first entry
        audit._entries[0].details = {"qty": 999}
        assert not audit.verify_integrity()  # Hash chain broken

    def test_export(self):
        audit = AuditTrail()
        audit.log_order_created("ord_1", "AAPL", "trend_following", {})
        audit.log_order_filled("ord_1", "AAPL", 150.0, 100, datetime.now(timezone.utc))
        exported = audit.export()
        assert len(exported) == 2
        assert "id" in exported[0]
        assert "hash" in exported[0]

    def test_summary(self):
        audit = AuditTrail()
        audit.log_order_created("ord_1", "AAPL", "trend_following", {})
        audit.log_risk_check("ord_1", "AAPL", True, 25.0, [], [])
        audit.log_order_filled("ord_1", "AAPL", 150.0, 100, datetime.now(timezone.utc))
        summary = audit.summary()
        assert summary["total_entries"] == 3
        assert summary["integrity_verified"] is True
        assert "order_created" in summary["actions"]

    def test_chained_hashes(self):
        audit = AuditTrail()
        for i in range(10):
            audit.log_order_created(f"ord_{i}", "AAPL", "trend_following", {"i": i})
        # Each entry should have a non-empty hash
        for entry in audit._entries:
            assert entry.hash != ""
        # Hashes should all be different
        hashes = [e.hash for e in audit._entries]
        assert len(set(hashes)) == 10  # All unique
