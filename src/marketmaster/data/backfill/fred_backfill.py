"""
FRED Historical Backfill — Ingest decades of macro data for MCEI computation.

This script fetches all FRED series needed for the MCEI from the earliest
available date through today, enabling walk-forward backtesting of regime
thresholds without look-ahead bias.

Usage:
    python -m marketmaster.data.backfill.fred_backfill [--start 1990-01-01] [--realtime]

With --realtime, fetches ALFRED vintage data (point-in-time) for each
observation. This is slower but necessary for bias-free backtesting.
"""

import asyncio
from datetime import date
from typing import Optional

import click

from marketmaster.config.settings import settings
from marketmaster.config.mcei_series import get_all_series_codes
from marketmaster.data.providers.fred import FredProvider
from marketmaster.data.ingestion import IngestionCoordinator
from marketmaster.db.session import get_session_factory


async def backfill_fred(
    start_date: date = date(1990, 1, 1),
    end_date: Optional[date] = None,
    realtime: bool = False,
    series_codes: Optional[list[str]] = None,
) -> dict:
    """
    Backfill FRED macro series data.

    Args:
        start_date: Earliest date to fetch (default: 1990-01-01)
        end_date: Latest date to fetch (default: today)
        realtime: If True, fetch ALFRED point-in-time vintage data
        series_codes: Specific series to fetch (default: all MCEI series)

    Returns summary of ingestion results.
    """
    if end_date is None:
        end_date = date.today()

    if series_codes is None:
        series_codes = get_all_series_codes()

    if not settings.fred_api_key:
        raise ValueError("FRED_API_KEY not configured. Set it in .env or environment.")

    provider = FredProvider(settings.fred_api_key)
    db = get_session_factory()()
    coordinator = IngestionCoordinator(db)

    results = {}
    total_observations = 0

    for code in series_codes:
        print(f"[FRED Backfill] Fetching {code} from {start_date} to {end_date}...")

        realtime_end = end_date if realtime else None

        result = await coordinator.ingest_macro_series(
            provider=provider,
            series_codes=[code],
            start=start_date,
            end=end_date,
            realtime_end=realtime_end,
        )

        results[code] = {
            "written": result.records_written,
            "skipped": result.records_skipped,
            "errors": result.errors,
            "status": result.status,
        }
        total_observations += result.records_written

        if result.errors:
            print(f"  ⚠ {len(result.errors)} errors: {result.errors[:2]}")
        else:
            print(f"  ✓ {result.records_written} observations written, {result.records_skipped} skipped")

    db.close()

    print(f"\n[FRED Backfill] Complete: {total_observations} total observations across {len(series_codes)} series")

    return {
        "total_observations": total_observations,
        "series_count": len(series_codes),
        "results": results,
    }


@click.command()
@click.option("--start", default="1990-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--realtime", is_flag=True, help="Fetch ALFRED point-in-time vintage data")
@click.option("--series", default=None, help="Comma-separated series codes (default: all MCEI series)")
def main(start: str, end: Optional[str], realtime: bool, series: Optional[str]):
    """Backfill FRED macro data for MarketMaster MCEI computation."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    series_codes = series.split(",") if series else None

    result = asyncio.run(backfill_fred(
        start_date=start_date,
        end_date=end_date,
        realtime=realtime,
        series_codes=series_codes,
    ))

    print(f"\nBackfill Summary:")
    print(f"  Total observations: {result['total_observations']}")
    print(f"  Series: {result['series_count']}")


if __name__ == "__main__":
    main()
