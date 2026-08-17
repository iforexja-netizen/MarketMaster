"""
MarketMaster Data Plane — Single Source of Truth Coordinator

Every agent, engine, and service reads data through the DataPlane.
No agent independently pulls data and reaches different conclusions.

Point-in-time semantics:
- macro_series: realtime_date parameter limits results to data available as-of that date
- fundamentals: filing_date is when data was actually available (vs report_date)
- ohlcv: adjusted_close and split_coefficient enable proper back-adjustment
"""

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import select, and_, desc
from sqlalchemy.orm import Session

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
)


class DataPlaneError(Exception):
    """Raised when the Data Plane encounters an error."""
    pass


class DataPlane:
    """
    The single entry point for all data access in MarketMaster.

    All agents, engines, and services use this class to read data.
    This ensures every consumer sees the same data with the same
    point-in-time semantics.
    """

    def __init__(self, db: Session):
        self.db = db

    # ========================================================================
    # SECURITY MASTER
    # ========================================================================

    def get_security_master(
        self,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        sector: Optional[str] = None,
        listing_status: str = "active",
    ) -> list[SecurityMaster]:
        """Query the canonical security master."""
        stmt = select(SecurityMaster).where(SecurityMaster.listing_status == listing_status)
        if symbol:
            stmt = stmt.where(SecurityMaster.symbol == symbol)
        if asset_class:
            stmt = stmt.where(SecurityMaster.asset_class == asset_class)
        if sector:
            stmt = stmt.where(SecurityMaster.sector == sector)
        return list(self.db.execute(stmt).scalars().all())

    def get_security_by_symbol(self, symbol: str) -> Optional[SecurityMaster]:
        """Get a single security by symbol."""
        stmt = select(SecurityMaster).where(SecurityMaster.symbol == symbol).limit(1)
        return self.db.execute(stmt).scalars().first()

    def get_security_by_id(self, security_id: int) -> Optional[SecurityMaster]:
        """Get a single security by ID."""
        return self.db.get(SecurityMaster, security_id)

    # ========================================================================
    # OHLCV DAILY
    # ========================================================================

    def get_ohlcv_daily(
        self,
        security_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[OhlcvDaily]:
        """
        Get daily OHLCV bars for a security.

        Point-in-time: adjusted_close and split_coefficient are included
        so backtests can properly adjust for corporate actions.
        """
        stmt = select(OhlcvDaily).where(OhlcvDaily.security_id == security_id)
        if start_date:
            stmt = stmt.where(OhlcvDaily.date >= start_date)
        if end_date:
            stmt = stmt.where(OhlcvDaily.date <= end_date)
        stmt = stmt.order_by(OhlcvDaily.date)
        return list(self.db.execute(stmt).scalars().all())

    def get_latest_price(self, security_id: int, as_of: Optional[date] = None) -> Optional[OhlcvDaily]:
        """Get the most recent OHLCV bar as-of a given date (or latest)."""
        stmt = select(OhlcvDaily).where(OhlcvDaily.security_id == security_id)
        if as_of:
            stmt = stmt.where(OhlcvDaily.date <= as_of)
        stmt = stmt.order_by(desc(OhlcvDaily.date)).limit(1)
        return self.db.execute(stmt).scalars().first()

    def get_latest_prices(self, as_of: Optional[date] = None) -> dict[int, OhlcvDaily]:
        """Get latest price for all securities. Returns {security_id: OhlcvDaily}."""
        # Use DISTINCT ON for PostgreSQL
        stmt = (
            select(OhlcvDaily)
            .where(OhlcvDaily.date <= as_of if as_of else True)
            .order_by(OhlcvDaily.security_id, desc(OhlcvDaily.date))
        )
        results = self.db.execute(stmt).scalars().all()
        seen: set[int] = set()
        prices: dict[int, OhlcvDaily] = {}
        for bar in results:
            if bar.security_id not in seen:
                seen.add(bar.security_id)
                prices[bar.security_id] = bar
        return prices

    # ========================================================================
    # OHLCV INTRADAY
    # ========================================================================

    def get_ohlcv_intraday(
        self,
        security_id: int,
        interval: str = "1m",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[OhlcvIntraday]:
        """Get intraday bars for a security."""
        stmt = select(OhlcvIntraday).where(
            and_(
                OhlcvIntraday.security_id == security_id,
                OhlcvIntraday.interval == interval,
            )
        )
        if start:
            stmt = stmt.where(OhlcvIntraday.timestamp >= start)
        if end:
            stmt = stmt.where(OhlcvIntraday.timestamp <= end)
        stmt = stmt.order_by(OhlcvIntraday.timestamp)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # CORPORATE ACTIONS
    # ========================================================================

    def get_corporate_actions(
        self,
        security_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[CorporateActions]:
        """Get corporate actions (splits, dividends) for a security."""
        stmt = select(CorporateActions).where(CorporateActions.security_id == security_id)
        if start_date:
            stmt = stmt.where(CorporateActions.ex_date >= start_date)
        if end_date:
            stmt = stmt.where(CorporateActions.ex_date <= end_date)
        stmt = stmt.order_by(CorporateActions.ex_date)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # FUNDAMENTALS (point-in-time via filing_date)
    # ========================================================================

    def get_fundamentals(
        self,
        security_id: int,
        statement_type: Optional[str] = None,
        realtime_date: Optional[date] = None,
    ) -> list[Fundamentals]:
        """
        Get fundamental data for a security.

        Point-in-time: if realtime_date is provided, only returns fundamentals
        whose filing_date <= realtime_date (i.e., data that was actually
        available on that date). This prevents look-ahead bias in backtests.
        """
        stmt = select(Fundamentals).where(Fundamentals.security_id == security_id)
        if statement_type:
            stmt = stmt.where(Fundamentals.statement_type == statement_type)
        if realtime_date:
            stmt = stmt.where(Fundamentals.filing_date <= realtime_date)
        stmt = stmt.order_by(desc(Fundamentals.report_date))
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # SEC FILINGS
    # ========================================================================

    def get_sec_filings(
        self,
        cik: Optional[str] = None,
        security_id: Optional[int] = None,
        form_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[SecFilings]:
        """Get SEC EDGAR filing metadata."""
        stmt = select(SecFilings)
        if cik:
            stmt = stmt.where(SecFilings.cik == cik)
        if security_id:
            stmt = stmt.where(SecFilings.security_id == security_id)
        if form_type:
            stmt = stmt.where(SecFilings.form_type == form_type)
        if start_date:
            stmt = stmt.where(SecFilings.filing_date >= start_date)
        if end_date:
            stmt = stmt.where(SecFilings.filing_date <= end_date)
        stmt = stmt.order_by(desc(SecFilings.filing_date))
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # MACRO SERIES (point-in-time via realtime_start/end)
    # ========================================================================

    def get_macro_series(
        self,
        series_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        realtime_date: Optional[date] = None,
    ) -> list[MacroSeries]:
        """
        Get macro series data from FRED/ALFRED.

        Point-in-time: if realtime_date is provided, returns only data
        whose realtime_start <= realtime_date (i.e., the value that was
        known on that date). This uses ALFRED vintage data to prevent
        look-ahead bias in backtests.

        Without realtime_date, returns the latest vintage (realtime_end
        is null or >= today) for each observation date.
        """
        stmt = select(MacroSeries).where(MacroSeries.series_code == series_code)
        if start_date:
            stmt = stmt.where(MacroSeries.observation_date >= start_date)
        if end_date:
            stmt = stmt.where(MacroSeries.observation_date <= end_date)
        if realtime_date:
            # Point-in-time: data that was available on realtime_date
            stmt = stmt.where(MacroSeries.realtime_start <= realtime_date)
            stmt = stmt.where(
                (MacroSeries.realtime_end.is_(None)) | (MacroSeries.realtime_end >= realtime_date)
            )
        stmt = stmt.order_by(MacroSeries.observation_date)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # ECONOMIC EVENTS
    # ========================================================================

    def get_economic_events(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        importance: Optional[str] = None,
    ) -> list[EconomicEvents]:
        """Get economic calendar events."""
        stmt = select(EconomicEvents)
        if start_date:
            stmt = stmt.where(EconomicEvents.event_date >= start_date)
        if end_date:
            stmt = stmt.where(EconomicEvents.event_date <= end_date)
        if importance:
            stmt = stmt.where(EconomicEvents.importance == importance)
        stmt = stmt.order_by(EconomicEvents.event_date)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # NEWS
    # ========================================================================

    def get_news(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[NewsItems]:
        """Get news items, optionally filtered by symbol."""
        stmt = select(NewsItems)
        if symbol:
            stmt = stmt.where(NewsItems.symbols.any(symbol))
        if start_date:
            stmt = stmt.where(NewsItems.news_date >= start_date)
        if end_date:
            stmt = stmt.where(NewsItems.news_date <= end_date)
        stmt = stmt.order_by(desc(NewsItems.news_date)).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # TRANSCRIPTS
    # ========================================================================

    def get_transcripts(
        self,
        security_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[Transcripts]:
        """Get earnings call transcripts for a security."""
        stmt = select(Transcripts).where(Transcripts.security_id == security_id)
        if start_date:
            stmt = stmt.where(Transcripts.call_date >= start_date)
        if end_date:
            stmt = stmt.where(Transcripts.call_date <= end_date)
        stmt = stmt.order_by(desc(Transcripts.call_date))
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # OPTION CHAINS
    # ========================================================================

    def get_option_chain(
        self,
        security_id: int,
        as_of_date: Optional[date] = None,
        expiration: Optional[date] = None,
    ) -> list[OptionChains]:
        """Get option chain data for a security."""
        stmt = select(OptionChains).where(OptionChains.security_id == security_id)
        if as_of_date:
            stmt = stmt.where(OptionChains.as_of_date == as_of_date)
        if expiration:
            stmt = stmt.where(OptionChains.expiration == expiration)
        stmt = stmt.order_by(OptionChains.expiration, OptionChains.strike)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # FEATURES
    # ========================================================================

    def get_features(
        self,
        security_id: int,
        feature_names: Optional[list[str]] = None,
        as_of_date: Optional[date] = None,
        feature_version: str = "v1",
    ) -> list[Features]:
        """Get computed features for a security."""
        stmt = select(Features).where(
            and_(
                Features.security_id == security_id,
                Features.feature_version == feature_version,
            )
        )
        if feature_names:
            stmt = stmt.where(Features.feature_name.in_(feature_names))
        if as_of_date:
            stmt = stmt.where(Features.as_of_date <= as_of_date)
        stmt = stmt.order_by(desc(Features.as_of_date))
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # SIGNALS
    # ========================================================================

    def get_signals(
        self,
        security_id: Optional[int] = None,
        signal_source: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100,
    ) -> list[Signals]:
        """Get trading signals with evidence."""
        stmt = select(Signals)
        if security_id:
            stmt = stmt.where(Signals.security_id == security_id)
        if signal_source:
            stmt = stmt.where(Signals.signal_source == signal_source)
        if start_date:
            stmt = stmt.where(Signals.as_of_date >= start_date)
        if end_date:
            stmt = stmt.where(Signals.as_of_date <= end_date)
        stmt = stmt.order_by(desc(Signals.as_of_date)).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # MCEI
    # ========================================================================

    def get_mcei(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[MceiHistory]:
        """Get MCEI history with component decomposition."""
        stmt = select(MceiHistory)
        if start_date:
            stmt = stmt.where(MceiHistory.as_of_date >= start_date)
        if end_date:
            stmt = stmt.where(MceiHistory.as_of_date <= end_date)
        stmt = stmt.order_by(desc(MceiHistory.as_of_date))
        return list(self.db.execute(stmt).scalars().all())

    def get_latest_mcei(self) -> Optional[MceiHistory]:
        """Get the most recent MCEI score and regime."""
        stmt = select(MceiHistory).order_by(desc(MceiHistory.as_of_date)).limit(1)
        return self.db.execute(stmt).scalars().first()

    # ========================================================================
    # REGIME
    # ========================================================================

    def get_regime_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[RegimeHistory]:
        """Get market regime classifications over time."""
        stmt = select(RegimeHistory)
        if start_date:
            stmt = stmt.where(RegimeHistory.as_of_date >= start_date)
        if end_date:
            stmt = stmt.where(RegimeHistory.as_of_date <= end_date)
        stmt = stmt.order_by(desc(RegimeHistory.as_of_date))
        return list(self.db.execute(stmt).scalars().all())

    def get_latest_regime(self) -> Optional[RegimeHistory]:
        """Get the most recent market regime classification."""
        stmt = select(RegimeHistory).order_by(desc(RegimeHistory.as_of_date)).limit(1)
        return self.db.execute(stmt).scalars().first()

    # ========================================================================
    # DECISIONS (immutable log — read only)
    # ========================================================================

    def get_decisions(
        self,
        security_id: Optional[int] = None,
        decision_type: Optional[str] = None,
        approved: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Decision]:
        """
        List decisions from the immutable decision log.
        This is read-only — decisions can only be appended via log_decision().
        """
        stmt = select(Decision)
        if security_id:
            stmt = stmt.where(Decision.security_id == security_id)
        if decision_type:
            stmt = stmt.where(Decision.decision_type == decision_type)
        if approved is not None:
            stmt = stmt.where(Decision.approved == approved)
        stmt = stmt.order_by(desc(Decision.timestamp)).limit(limit).offset(offset)
        return list(self.db.execute(stmt).scalars().all())

    def get_decision(self, decision_id: int) -> Optional[Decision]:
        """Get a single decision with full evidence."""
        return self.db.get(Decision, decision_id)

    # ========================================================================
    # TRADES
    # ========================================================================

    def get_trades(
        self,
        security_id: Optional[int] = None,
        status: Optional[str] = None,
        is_paper: bool = True,
        limit: int = 100,
    ) -> list[Trade]:
        """Get trade records."""
        stmt = select(Trade).where(Trade.is_paper == is_paper)
        if security_id:
            stmt = stmt.where(Trade.security_id == security_id)
        if status:
            stmt = stmt.where(Trade.status == status)
        stmt = stmt.order_by(desc(Trade.placed_at)).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    # ========================================================================
    # PORTFOLIO
    # ========================================================================

    def get_latest_portfolio(self, is_paper: bool = True) -> Optional[PortfolioSnapshot]:
        """Get the latest portfolio snapshot."""
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.is_paper == is_paper)
            .order_by(desc(PortfolioSnapshot.as_of_date))
            .limit(1)
        )
        return self.db.execute(stmt).scalars().first()

    # ========================================================================
    # RISK METRICS
    # ========================================================================

    def get_latest_risk_metrics(self, as_of: Optional[date] = None) -> list[RiskMetric]:
        """Get the latest risk metrics."""
        stmt = select(RiskMetric)
        if as_of:
            stmt = stmt.where(RiskMetric.as_of_date == as_of)
        else:
            # Get the most recent date
            latest = self.db.execute(
                select(RiskMetric.as_of_date).order_by(desc(RiskMetric.as_of_date)).limit(1)
            ).scalar()
            if latest:
                stmt = stmt.where(RiskMetric.as_of_date == latest)
        return list(self.db.execute(stmt).scalars().all())
