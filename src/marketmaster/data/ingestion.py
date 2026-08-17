"""
Ingestion Coordinator — Orchestrates data fetching and DB writing.

Handles:
- Provider → DB mapping (symbol → security_master.id resolution)
- Idempotent ingestion (skip if record already exists)
- Ingestion logging (for observability and idempotency)
- Batch writing for performance
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketmaster.db.models import (
    SecurityMaster,
    OhlcvDaily,
    CorporateActions,
    Fundamentals,
    SecFilings,
    MacroSeries,
    IngestionLog,
)
from marketmaster.data.providers.base import DataProvider
from marketmaster.data.providers.alpaca import AlpacaProvider
from marketmaster.data.providers.fred import FredProvider
from marketmaster.data.providers.sec import SecEdgarProvider


@dataclass
class IngestionResult:
    """Result of an ingestion run."""
    run_id: str
    provider: str
    data_type: str
    scope: str
    records_written: int = 0
    records_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def status(self) -> str:
        if self.errors and self.records_written == 0:
            return "failed"
        if self.errors:
            return "partial"
        return "success"


class IngestionCoordinator:
    """
    Coordinates data ingestion from providers into the Data Plane.

    All ingestion is idempotent: if a record already exists (by unique key),
    it is skipped rather than duplicated or overwritten.
    """

    def __init__(self, db: Session):
        self.db = db

    def _resolve_security_id(self, symbol: str) -> Optional[int]:
        """Resolve a ticker symbol to security_master.id."""
        stmt = select(SecurityMaster.id).where(SecurityMaster.symbol == symbol).limit(1)
        return self.db.execute(stmt).scalar()

    def _log_ingestion(self, result: IngestionResult) -> None:
        """Log an ingestion run to the database."""
        entry = IngestionLog(
            run_id=uuid.UUID(result.run_id),
            provider=result.provider,
            data_type=result.data_type,
            scope=result.scope,
            records_written=result.records_written,
            records_skipped=result.records_skipped,
            status=result.status,
            error_message="; ".join(result.errors) if result.errors else None,
            started_at=result.started_at,
            completed_at=result.completed_at,
        )
        self.db.add(entry)
        self.db.commit()

    # ── OHLCV Daily ───────────────────────────────────────────────────────────

    async def ingest_ohlcv_daily(
        self,
        provider: AlpacaProvider,
        symbols: list[str],
        start: date,
        end: date,
    ) -> IngestionResult:
        """Ingest daily OHLCV bars for a list of symbols."""
        run_id = str(uuid.uuid4())
        scope = f"symbols={','.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}, range={start} to {end}"
        result = IngestionResult(
            run_id=run_id, provider=provider.name, data_type="ohlcv_daily", scope=scope,
        )

        for symbol in symbols:
            security_id = self._resolve_security_id(symbol)
            if not security_id:
                result.errors.append(f"Unknown symbol: {symbol}")
                continue

            try:
                bars = await provider.fetch_ohlcv_daily(symbol, start, end)
                for bar in bars:
                    # Check if record exists (idempotent)
                    existing = self.db.execute(
                        select(OhlcvDaily).where(
                            OhlcvDaily.security_id == security_id,
                            OhlcvDaily.date == bar["date"],
                        )
                    ).scalars().first()

                    if existing:
                        result.records_skipped += 1
                        continue

                    record = OhlcvDaily(
                        security_id=security_id,
                        date=bar["date"],
                        open=bar.get("open"),
                        high=bar.get("high"),
                        low=bar.get("low"),
                        close=bar.get("close"),
                        volume=bar.get("volume"),
                        vwap=bar.get("vwap"),
                        source=provider.name,
                    )
                    self.db.add(record)
                    result.records_written += 1

                self.db.commit()

            except Exception as e:
                result.errors.append(f"{symbol}: {str(e)}")
                self.db.rollback()

        result.completed_at = datetime.now(timezone.utc)
        self._log_ingestion(result)
        return result

    # ── Macro Series ─────────────────────────────────────────────────────────

    async def ingest_macro_series(
        self,
        provider: FredProvider,
        series_codes: list[str],
        start: date,
        end: date,
        realtime_end: Optional[date] = None,
    ) -> IngestionResult:
        """Ingest FRED macro series data."""
        run_id = str(uuid.uuid4())
        scope = f"series={','.join(series_codes[:5])}{'...' if len(series_codes) > 5 else ''}, range={start} to {end}"
        result = IngestionResult(
            run_id=run_id, provider=provider.name, data_type="macro_series", scope=scope,
        )

        for code in series_codes:
            try:
                observations = await provider.fetch_macro_series(code, start, end, realtime_end)
                for obs in observations:
                    # Check if exists (idempotent)
                    stmt = select(MacroSeries).where(
                        MacroSeries.series_code == obs["series_code"],
                        MacroSeries.observation_date == obs["observation_date"],
                    )
                    if obs.get("realtime_start"):
                        stmt = stmt.where(MacroSeries.realtime_start == obs["realtime_start"])

                    existing = self.db.execute(stmt).scalars().first()
                    if existing:
                        result.records_skipped += 1
                        continue

                    record = MacroSeries(
                        series_code=obs["series_code"],
                        observation_date=obs["observation_date"],
                        value=obs["value"],
                        realtime_start=obs.get("realtime_start"),
                        realtime_end=obs.get("realtime_end"),
                        source=provider.name,
                    )
                    self.db.add(record)
                    result.records_written += 1

                self.db.commit()

            except Exception as e:
                result.errors.append(f"{code}: {str(e)}")
                self.db.rollback()

        result.completed_at = datetime.now(timezone.utc)
        self._log_ingestion(result)
        return result

    # ── SEC Filings ──────────────────────────────────────────────────────────

    async def ingest_sec_filings(
        self,
        provider: SecEdgarProvider,
        cik: str,
        form_types: list[str],
        start: date,
        end: date,
    ) -> IngestionResult:
        """Ingest SEC EDGAR filing metadata."""
        run_id = str(uuid.uuid4())
        scope = f"cik={cik}, forms={','.join(form_types)}, range={start} to {end}"
        result = IngestionResult(
            run_id=run_id, provider=provider.name, data_type="sec_filings", scope=scope,
        )

        try:
            filings = await provider.fetch_filings(cik, form_types, start, end)

            # Try to resolve CIK to security_id
            security_id = self.db.execute(
                select(SecurityMaster.id).where(SecurityMaster.cik == cik)
            ).scalar()

            for filing in filings:
                existing = self.db.execute(
                    select(SecFilings).where(
                        SecFilings.cik == filing["cik"],
                        SecFilings.accession_no == filing["accession_no"],
                    )
                ).scalars().first()

                if existing:
                    result.records_skipped += 1
                    continue

                record = SecFilings(
                    security_id=security_id,
                    cik=filing["cik"],
                    accession_no=filing["accession_no"],
                    filing_date=filing["filing_date"],
                    form_type=filing["form_type"],
                    description=filing.get("description"),
                    primary_document=filing.get("primary_document"),
                    filing_url=filing.get("filing_url"),
                )
                self.db.add(record)
                result.records_written += 1

            self.db.commit()

        except Exception as e:
            result.errors.append(str(e))
            self.db.rollback()

        result.completed_at = datetime.now(timezone.utc)
        self._log_ingestion(result)
        return result

    # ── Fundamentals ──────────────────────────────────────────────────────────

    async def ingest_fundamentals(
        self,
        provider: SecEdgarProvider,
        cik: str,
    ) -> IngestionResult:
        """Ingest structured fundamentals from SEC XBRL data."""
        run_id = str(uuid.uuid4())
        scope = f"cik={cik}"
        result = IngestionResult(
            run_id=run_id, provider=provider.name, data_type="fundamentals", scope=scope,
        )

        security_id = self.db.execute(
            select(SecurityMaster.id).where(SecurityMaster.cik == cik)
        ).scalar()

        if not security_id:
            result.errors.append(f"No security found for CIK {cik}")
            result.completed_at = datetime.now(timezone.utc)
            self._log_ingestion(result)
            return result

        try:
            fundamentals = await provider.fetch_fundamentals(cik)

            for fund in fundamentals:
                existing = self.db.execute(
                    select(Fundamentals).where(
                        Fundamentals.security_id == security_id,
                        Fundamentals.report_date == fund["report_date"],
                        Fundamentals.statement_type == fund["statement_type"],
                    )
                ).scalars().first()

                if existing:
                    result.records_skipped += 1
                    continue

                record = Fundamentals(
                    security_id=security_id,
                    report_date=fund["report_date"],
                    period_type=fund.get("period_type", "annual"),
                    statement_type=fund["statement_type"],
                    items=fund["items"],
                    source=provider.name,
                    filing_date=fund.get("filing_date"),
                )
                self.db.add(record)
                result.records_written += 1

            self.db.commit()

        except Exception as e:
            result.errors.append(str(e))
            self.db.rollback()

        result.completed_at = datetime.now(timezone.utc)
        self._log_ingestion(result)
        return result
