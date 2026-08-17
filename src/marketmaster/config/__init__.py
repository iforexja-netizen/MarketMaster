"""
MarketMaster configuration package.
"""

from marketmaster.config.settings import settings
from marketmaster.config.mcei_series import (
    MCEI_COMPONENTS,
    MCEIComponent,
    MCEI_REGIME_THRESHOLDS,
    MARKET_REGIMES,
    get_all_series_codes,
    get_total_weight,
)

__all__ = [
    "settings",
    "MCEI_COMPONENTS",
    "MCEIComponent",
    "MCEI_REGIME_THRESHOLDS",
    "MARKET_REGIMES",
    "get_all_series_codes",
    "get_total_weight",
]
