"""
Abstract base class for all MarketMaster data providers.

Providers encapsulate external API calls. They return normalized dicts
matching DB column names (security_id is resolved by the ingestion layer).
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Optional


class DataProvider(ABC):
    """Base class for all data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'alpaca', 'fred', 'sec_edgar')."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider API is reachable and credentials are valid."""
        ...

    # ── Market Data ──────────────────────────────────────────────────────────

    async def fetch_ohlcv_daily(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Fetch daily OHLCV bars. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support daily OHLCV")

    async def fetch_ohlcv_intraday(
        self, symbol: str, start: datetime, end: datetime, interval: str = "1m"
    ) -> list[dict[str, Any]]:
        """Fetch intraday OHLCV bars. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support intraday OHLCV")

    async def fetch_corporate_actions(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Fetch corporate actions (splits, dividends). Returns normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support corporate actions")

    async def fetch_option_chain(
        self, symbol: str, as_of_date: date
    ) -> list[dict[str, Any]]:
        """Fetch option chain snapshot. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support option chains")

    # ── Fundamentals & Filings ─────────────────────────────────────────────────

    async def fetch_fundamentals(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch fundamental data. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support fundamentals")

    async def fetch_filings(
        self, cik: str, form_types: list[str], start: date, end: date
    ) -> list[dict[str, Any]]:
        """Fetch SEC filing metadata. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support filings")

    # ── Macro ─────────────────────────────────────────────────────────────────

    async def fetch_macro_series(
        self,
        series_id: str,
        start: date,
        end: date,
        realtime_end: Optional[date] = None,
    ) -> list[dict[str, Any]]:
        """Fetch macro series data. Returns list of normalized dicts."""
        raise NotImplementedError(f"{self.name} does not support macro series")
