"""
Alpaca Markets Data Provider

Fetches market data from Alpaca's REST API:
- Daily and intraday OHLCV bars
- Corporate actions (splits, dividends)
- Option chains

All API-specific logic stays inside this adapter. Returns normalized dicts
matching DB column names.
"""

import os
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx

from marketmaster.data.providers.base import DataProvider


class AlpacaProvider(DataProvider):
    """Alpaca Markets data provider for market data and paper trading."""

    BASE_URL = "https://data.alpaca.markets/v2"
    PAPER_TRADING_URL = "https://paper-api.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        data_url: str = BASE_URL,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.data_url = data_url

    @property
    def name(self) -> str:
        return "alpaca"

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    async def health_check(self) -> bool:
        """Check if Alpaca API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.PAPER_TRADING_URL}/v2/account",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def fetch_ohlcv_daily(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """
        Fetch daily OHLCV bars from Alpaca.

        Returns list of normalized dicts with keys:
        date, open, high, low, close, volume, vwap
        """
        params = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "all",  # split + dividend adjusted
        }
        bars: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.data_url}/stocks/{symbol}/bars"
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()

            for bar in data.get("bars", []):
                t = datetime.fromisoformat(bar["t"])
                bars.append({
                    "date": t.date(),
                    "open": float(bar.get("o", 0)),
                    "high": float(bar.get("h", 0)),
                    "low": float(bar.get("l", 0)),
                    "close": float(bar.get("c", 0)),
                    "volume": int(bar.get("v", 0)),
                    "vwap": float(bar.get("vw", 0)) if bar.get("vw") else None,
                })

        return bars

    async def fetch_ohlcv_intraday(
        self, symbol: str, start: datetime, end: datetime, interval: str = "1m"
    ) -> list[dict[str, Any]]:
        """
        Fetch intraday OHLCV bars from Alpaca.

        interval: '1m', '5m', '15m', '1h'
        """
        timeframe_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        tf = timeframe_map.get(interval, "1Min")

        params = {
            "timeframe": tf,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
        }
        bars: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.data_url}/stocks/{symbol}/bars"
            resp = await client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()

            for bar in data.get("bars", []):
                t = datetime.fromisoformat(bar["t"])
                bars.append({
                    "timestamp": t,
                    "interval": interval,
                    "open": float(bar.get("o", 0)),
                    "high": float(bar.get("h", 0)),
                    "low": float(bar.get("l", 0)),
                    "close": float(bar.get("c", 0)),
                    "volume": int(bar.get("v", 0)),
                    "vwap": float(bar.get("vw", 0)) if bar.get("vw") else None,
                })

        return bars

    async def fetch_corporate_actions(
        self, symbol: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """
        Fetch corporate actions from Alpaca.

        Returns normalized dicts with: ex_date, action_type, value
        """
        actions: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Alpaca corporate announcements endpoint
            url = f"{self.data_url}/corporate_announcements"
            params = {
                "since": start.isoformat(),
                "until": end.isoformat(),
                "symbol": symbol,
            }
            resp = await client.get(url, headers=self._headers(), params=params)
            if resp.status_code != 200:
                return actions
            data = resp.json()

            for item in data if isinstance(data, list) else []:
                action_type = item.get("ca_type", "")
                actions.append({
                    "ex_date": date.fromisoformat(item.get("ex_date", "")),
                    "action_type": action_type,
                    "description": item.get("ca_type_description", ""),
                    "value": float(item.get("ratio", 0)) if item.get("ratio") else
                             float(item.get("cash_amount", 0)) if item.get("cash_amount") else None,
                })

        return actions

    async def fetch_option_chain(
        self, symbol: str, as_of_date: date
    ) -> list[dict[str, Any]]:
        """
        Fetch option chain from Alpaca (if available).

        Returns normalized dicts with: expiration, strike, option_type,
        bid, ask, volume, open_interest, iv, delta, gamma, theta, vega, rho
        """
        # Alpaca options API may not be available for all accounts
        # This is a placeholder that can be extended when the API is ready
        raise NotImplementedError(
            "Alpaca options chain requires the Options Data subscription. "
            "Implement when subscription is active."
        )
