"""
MarketMaster execution package.

Phase 5: Order lifecycle, broker integration, position monitoring, and audit trail.
"""

from marketmaster.execution.broker import (
    AlpacaPaperBroker, BrokerOrder, BrokerPosition, AccountState,
    OrderStatus, BrokerOrderSide, BrokerOrderType,
)
from marketmaster.execution.lifecycle import (
    OrderLifecycleManager, ManagedOrder, LifecycleState, LifecycleResult,
)
from marketmaster.execution.monitor import (
    PositionMonitor, PositionAlert, MonitoringResult, AlertType, AlertAction,
)
from marketmaster.execution.audit import AuditTrail, AuditEntry, AuditActionType

__all__ = [
    "AlpacaPaperBroker", "BrokerOrder", "BrokerPosition", "AccountState",
    "OrderStatus", "BrokerOrderSide", "BrokerOrderType",
    "OrderLifecycleManager", "ManagedOrder", "LifecycleState", "LifecycleResult",
    "PositionMonitor", "PositionAlert", "MonitoringResult", "AlertType", "AlertAction",
    "AuditTrail", "AuditEntry", "AuditActionType",
]
