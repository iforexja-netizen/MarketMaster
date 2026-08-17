"""
FRED (Federal Reserve Economic Data) Provider

Fetches macro economic data from the FRED API.
Supports ALFRED point-in-time vintage data via realtime_start/realtime_end.

This is critical for the MCEI engine — all macro components come from here.
"""

from datetime import date
from typing import Any, Optional

import httpx

from marketmaster.data.providers.base import DataProvider


class FredProvider(DataProvider):
    """FRED/ALFRED data provider for macro economic series."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "fred"

    async def health_check(self) -> bool:
        """Check if FRED API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/series",
                    params={
                        "series_id": "DGS10",
                        "api_key": self.api_key,
                        "file_type": "json",
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def fetch_macro_series(
        self,
        series_id: str,
        start: date,
        end: date,
        realtime_end: Optional[date] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch macro series observations from FRED.

        If realtime_end is provided, uses ALFRED point-in-time vintage data:
        returns the value that was known as of realtime_end. This is
        essential for backtesting without look-ahead bias.

        Returns normalized dicts with:
        series_code, observation_date, value, realtime_start, realtime_end
        """
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }

        # ALFRED point-in-time: if realtime_end is set, fetch vintage data
        if realtime_end is not None:
            params["realtime_start"] = "1776-07-04"  # all vintages
            params["realtime_end"] = realtime_end.isoformat()

        observations: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            for obs in data.get("observations", []):
                value_str = obs.get("value", ".")
                # FRED uses "." for missing values
                if value_str == "." or value_str is None:
                    continue

                observations.append({
                    "series_code": series_id,
                    "observation_date": date.fromisoformat(obs["date"]),
                    "value": float(value_str),
                    "realtime_start": date.fromisoformat(obs["realtime_start"]) if realtime_end else None,
                    "realtime_end": date.fromisoformat(obs["realtime_end"]) if realtime_end else None,
                })

        return observations

    async def fetch_series_info(self, series_id: str) -> dict[str, Any]:
        """Fetch metadata about a FRED series (frequency, units, title)."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.BASE_URL}/series",
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            series_list = data.get("seriess", [])
            if series_list:
                s = series_list[0]
                return {
                    "series_code": series_id,
                    "series_name": s.get("title"),
                    "frequency": s.get("frequency_short"),
                    "units": s.get("units_short"),
                    "seasonally_adj": s.get("seasonal_adjustment") == "Seasonally Adjusted",
                }
            return {}
