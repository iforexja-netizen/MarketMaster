"""
MarketMaster Database Package

Provides SQLAlchemy 2.0 ORM models, session management, and the
immutable decision log utility.
"""

from marketmaster.db.base import Base
from marketmaster.db.session import (
    get_db,
    get_async_db,
    get_engine,
    get_async_engine,
    get_session_factory,
    get_async_session_factory,
)

# Import all models so they are registered with Base.metadata
from marketmaster.db.models import (
    SecurityMaster,
    OhlcvDaily,
    OhlcvIntraday,
    CorporateActions,
    Fundamentals,
    SecFilings,
    MacroSeries,
    EconomicEvents,
    NewsItems,
    Transcripts,
    OptionChains,
    Features,
    Signals,
    MceiHistory,
    RegimeHistory,
    Decision,
    Trade,
    PortfolioSnapshot,
    RiskMetric,
    DataQualityLog,
    IngestionLog,
    MceiConfig,
)

from marketmaster.db.decision_log import log_decision, verify_chain_integrity

__all__ = [
    "Base",
    "get_db",
    "get_async_db",
    "get_engine",
    "get_async_engine",
    "get_session_factory",
    "get_async_session_factory",
    # Models
    "SecurityMaster",
    "OhlcvDaily",
    "OhlcvIntraday",
    "CorporateActions",
    "Fundamentals",
    "SecFilings",
    "MacroSeries",
    "EconomicEvents",
    "NewsItems",
    "Transcripts",
    "OptionChains",
    "Features",
    "Signals",
    "MceiHistory",
    "RegimeHistory",
    "Decision",
    "Trade",
    "PortfolioSnapshot",
    "RiskMetric",
    "DataQualityLog",
    "IngestionLog",
    "MceiConfig",
    # Decision log
    "log_decision",
    "verify_chain_integrity",
]
