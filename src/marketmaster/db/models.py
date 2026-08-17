"""
MarketMaster SQLAlchemy 2.0 ORM Models

Matches db/schema.sql exactly. All 21 tables + MCEI config.
Uses Mapped/mapped_column style (SQLAlchemy 2.0+).
"""

from datetime import date, datetime, time
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from marketmaster.db.base import Base


# ============================================================================
# 1. SECURITY MASTER
# ============================================================================
class SecurityMaster(Base):
    __tablename__ = "security_master"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(512))
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False, default="equity")
    exchange: Mapped[Optional[str]] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    sector: Mapped[Optional[str]] = mapped_column(String(128))
    industry: Mapped[Optional[str]] = mapped_column(String(128))
    sub_industry: Mapped[Optional[str]] = mapped_column(String(128))
    cik: Mapped[Optional[str]] = mapped_column(String(16))
    figi: Mapped[Optional[str]] = mapped_column(String(20))
    isin: Mapped[Optional[str]] = mapped_column(String(16))
    cusip: Mapped[Optional[str]] = mapped_column(String(12))
    composite_figi: Mapped[Optional[str]] = mapped_column(String(20))
    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger)
    listing_status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    listing_date: Mapped[Optional[date]] = mapped_column(Date)
    delisting_date: Mapped[Optional[date]] = mapped_column(Date)
    tick_size: Mapped[Optional[Any]] = mapped_column(Numeric(12, 6))
    currency_primary: Mapped[Optional[str]] = mapped_column(String(8), default="USD")
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", name="security_master_symbol_exchange_key"),
        Index("idx_sec_master_symbol", "symbol"),
        Index("idx_sec_master_asset_class", "asset_class"),
        Index("idx_sec_master_sector", "sector"),
        Index("idx_sec_master_cik", "cik"),
        Index("idx_sec_master_status", "listing_status"),
    )

    # Relationships
    ohlcv_daily: Mapped[list["OhlcvDaily"]] = relationship(back_populates="security")
    ohlcv_intraday: Mapped[list["OhlcvIntraday"]] = relationship(back_populates="security")
    corporate_actions: Mapped[list["CorporateActions"]] = relationship(back_populates="security")
    fundamentals: Mapped[list["Fundamentals"]] = relationship(back_populates="security")
    sec_filings: Mapped[list["SecFilings"]] = relationship(back_populates="security")
    transcripts: Mapped[list["Transcripts"]] = relationship(back_populates="security")
    option_chains: Mapped[list["OptionChains"]] = relationship(back_populates="security")
    features: Mapped[list["Features"]] = relationship(back_populates="security")
    signals: Mapped[list["Signals"]] = relationship(back_populates="security")
    trades: Mapped[list["Trade"]] = relationship(back_populates="security")


# ============================================================================
# 2. OHLCV DAILY
# ============================================================================
class OhlcvDaily(Base):
    __tablename__ = "ohlcv_daily"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    high: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    low: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    close: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    adjusted_close: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    dividend_amount: Mapped[Any] = mapped_column(Numeric(20, 6), default=0)
    split_coefficient: Mapped[Any] = mapped_column(Numeric(20, 10), default=1)
    vwap: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="ohlcv_daily")

    __table_args__ = (
        UniqueConstraint("security_id", "date", name="ohlcv_daily_security_id_date_key"),
        Index("idx_ohlcv_daily_sec_date", "security_id", "date"),
        Index("idx_ohlcv_daily_date", "date"),
    )


# ============================================================================
# 3. OHLCV INTRADAY
# ============================================================================
class OhlcvIntraday(Base):
    __tablename__ = "ohlcv_intraday"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    open: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    high: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    low: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    close: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    vwap: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="ohlcv_intraday")

    __table_args__ = (
        UniqueConstraint("security_id", "timestamp", "interval", name="ohlcv_intra_sec_ts_interval_key"),
        Index("idx_ohlcv_intra_sec_ts", "security_id", "timestamp"),
        Index("idx_ohlcv_intra_interval", "interval"),
    )


# ============================================================================
# 4. CORPORATE ACTIONS
# ============================================================================
class CorporateActions(Base):
    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(512))
    value: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    value_fractional: Mapped[Optional[Any]] = mapped_column(Numeric(20, 10))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="corporate_actions")

    __table_args__ = (
        UniqueConstraint("security_id", "ex_date", "action_type", name="corp_actions_sec_date_type_key"),
        Index("idx_corp_actions_sec_date", "security_id", "ex_date"),
    )


# ============================================================================
# 5. FUNDAMENTALS
# ============================================================================
class Fundamentals(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    fiscal_year: Mapped[Optional[int]] = mapped_column(Integer)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer)
    period_type: Mapped[str] = mapped_column(String(16), nullable=False)
    statement_type: Mapped[str] = mapped_column(String(32), nullable=False)
    items: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="sec_edgar")
    filing_date: Mapped[Optional[date]] = mapped_column(Date)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="fundamentals")

    __table_args__ = (
        UniqueConstraint("security_id", "report_date", "period_type", "statement_type", name="fundamentals_key"),
        Index("idx_fundamentals_sec_date", "security_id", "report_date"),
        Index("idx_fundamentals_filing", "filing_date"),
    )


# ============================================================================
# 6. SEC FILINGS
# ============================================================================
class SecFilings(Base):
    __tablename__ = "sec_filings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=True)
    cik: Mapped[str] = mapped_column(String(16), nullable=False)
    accession_no: Mapped[str] = mapped_column(String(32), nullable=False)
    filing_date: Mapped[date] = mapped_column(Date, nullable=False)
    form_type: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    primary_document: Mapped[Optional[str]] = mapped_column(String(256))
    filing_url: Mapped[Optional[str]] = mapped_column(String(512))
    parsed: Mapped[bool] = mapped_column(Boolean, default=False)
    parsed_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped[Optional["SecurityMaster"]] = relationship(back_populates="sec_filings")

    __table_args__ = (
        UniqueConstraint("cik", "accession_no", name="sec_filings_cik_accession_key"),
        Index("idx_sec_filings_cik_date", "cik", "filing_date"),
        Index("idx_sec_filings_form", "form_type"),
    )


# ============================================================================
# 7. MACRO SERIES
# ============================================================================
class MacroSeries(Base):
    __tablename__ = "macro_series"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    series_code: Mapped[str] = mapped_column(String(64), nullable=False)
    series_name: Mapped[Optional[str]] = mapped_column(String(256))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="fred")
    frequency: Mapped[Optional[str]] = mapped_column(String(16))
    units: Mapped[Optional[str]] = mapped_column(String(64))
    seasonally_adj: Mapped[bool] = mapped_column(Boolean, default=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    value: Mapped[Optional[Any]] = mapped_column(Numeric(24, 8))
    realtime_start: Mapped[Optional[date]] = mapped_column(Date)
    realtime_end: Mapped[Optional[date]] = mapped_column(Date)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("macro_series_key",
              "series_code", "observation_date",
              text("COALESCE(realtime_start, DATE '1900-01-01')"),
              unique=True),
        Index("idx_macro_series_code_date", "series_code", "observation_date"),
        Index("idx_macro_series_realtime", "series_code", "realtime_start", "realtime_end"),
    )


# ============================================================================
# 8. ECONOMIC EVENTS
# ============================================================================
class EconomicEvents(Base):
    __tablename__ = "economic_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_time: Mapped[Optional[time]] = mapped_column(Time)
    country: Mapped[str] = mapped_column(String(8), nullable=False, default="US")
    event_name: Mapped[str] = mapped_column(String(256), nullable=False)
    importance: Mapped[str] = mapped_column(String(16), default="medium")
    actual: Mapped[Optional[Any]] = mapped_column(Numeric(24, 8))
    forecast: Mapped[Optional[Any]] = mapped_column(Numeric(24, 8))
    previous: Mapped[Optional[Any]] = mapped_column(Numeric(24, 8))
    actual_unit: Mapped[Optional[str]] = mapped_column(String(32))
    source: Mapped[Optional[str]] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("event_date", "event_name", "country", name="econ_events_key"),
        Index("idx_econ_events_date", "event_date"),
    )


# ============================================================================
# 9. NEWS ITEMS
# ============================================================================
class NewsItems(Base):
    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    news_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(512))
    symbols: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String(32)))
    sentiment_score: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(16))
    relevance_score: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    raw: Mapped[dict] = mapped_column(JSONB, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_news_date", "news_date"),
        Index("idx_news_symbols", "symbols", postgresql_using="gin"),
    )


# ============================================================================
# 10. TRANSCRIPTS
# ============================================================================
class Transcripts(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    call_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fiscal_year: Mapped[Optional[int]] = mapped_column(Integer)
    fiscal_quarter: Mapped[Optional[int]] = mapped_column(Integer)
    transcript_type: Mapped[str] = mapped_column(String(32), default="earnings_call")
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    parsed: Mapped[dict] = mapped_column(JSONB, default=dict)
    source: Mapped[Optional[str]] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="transcripts")

    __table_args__ = (
        UniqueConstraint("security_id", "call_date", name="transcripts_sec_call_date_key"),
        Index("idx_transcripts_sec_date", "security_id", "call_date"),
    )


# ============================================================================
# 11. OPTION CHAINS
# ============================================================================
class OptionChains(Base):
    __tablename__ = "option_chains"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiration: Mapped[date] = mapped_column(Date, nullable=False)
    strike: Mapped[Any] = mapped_column(Numeric(12, 4), nullable=False)
    option_type: Mapped[str] = mapped_column(String(4), nullable=False)
    symbol: Mapped[Optional[str]] = mapped_column(String(64))
    bid: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    ask: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    last: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    open_interest: Mapped[Optional[int]] = mapped_column(BigInteger)
    iv: Mapped[Optional[Any]] = mapped_column(Numeric(10, 6))
    delta: Mapped[Optional[Any]] = mapped_column(Numeric(10, 6))
    gamma: Mapped[Optional[Any]] = mapped_column(Numeric(12, 8))
    theta: Mapped[Optional[Any]] = mapped_column(Numeric(12, 8))
    vega: Mapped[Optional[Any]] = mapped_column(Numeric(12, 8))
    rho: Mapped[Optional[Any]] = mapped_column(Numeric(12, 8))
    underlying_price: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alpaca")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="option_chains")

    __table_args__ = (
        UniqueConstraint("security_id", "as_of_date", "expiration", "strike", "option_type", name="option_chains_key"),
        Index("idx_options_sec_date", "security_id", "as_of_date"),
        Index("idx_options_expiry", "expiration"),
    )


# ============================================================================
# 12. FEATURES
# ============================================================================
class Features(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_name: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_value: Mapped[Optional[Any]] = mapped_column(Numeric(24, 10))
    feature_category: Mapped[Optional[str]] = mapped_column(String(32))
    feature_version: Mapped[str] = mapped_column(String(16), default="v1")
    lookback_days: Mapped[Optional[int]] = mapped_column(Integer)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="features")

    __table_args__ = (
        UniqueConstraint("security_id", "as_of_date", "feature_name", "feature_version", name="features_key"),
        Index("idx_features_sec_date", "security_id", "as_of_date"),
        Index("idx_features_name", "feature_name"),
        Index("idx_features_category", "feature_category"),
    )


# ============================================================================
# 13. SIGNALS
# ============================================================================
class Signals(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_source: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    strength: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    score: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    confidence: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    strategy: Mapped[Optional[str]] = mapped_column(String(64))
    regime: Mapped[Optional[str]] = mapped_column(String(32))
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    data_quality: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    security: Mapped["SecurityMaster"] = relationship(back_populates="signals")

    __table_args__ = (
        Index("idx_signals_sec_date", "security_id", "as_of_date"),
        Index("idx_signals_source", "signal_source"),
        Index("idx_signals_date", "as_of_date"),
    )


# ============================================================================
# 14. MCEI HISTORY
# ============================================================================
class MceiHistory(Base):
    __tablename__ = "mcei_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    score: Mapped[Any] = mapped_column(Numeric(6, 2), nullable=False)
    regime: Mapped[str] = mapped_column(String(32), nullable=False)
    components: Mapped[dict] = mapped_column(JSONB, nullable=False)
    weights_version: Mapped[str] = mapped_column(String(16), default="v1")
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_mcei_date", "as_of_date"),
    )


# ============================================================================
# 15. REGIME HISTORY
# ============================================================================
class RegimeHistory(Base):
    __tablename__ = "regime_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    regime: Mapped[str] = mapped_column(String(32), nullable=False)
    prev_regime: Mapped[Optional[str]] = mapped_column(String(32))
    transition_date: Mapped[Optional[date]] = mapped_column(Date)
    confidence: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    regime_version: Mapped[str] = mapped_column(String(16), default="v1")
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_regime_date", "as_of_date"),
    )


# ============================================================================
# 16. DECISIONS (IMMUTABLE — APPEND ONLY)
# ============================================================================
class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    decision_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prev_hash: Mapped[Optional[str]] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    security_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32))
    decision_type: Mapped[str] = mapped_column(String(32), nullable=False)
    strategy: Mapped[Optional[str]] = mapped_column(String(64))
    regime: Mapped[Optional[str]] = mapped_column(String(32))
    approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    score: Mapped[Optional[Any]] = mapped_column(Numeric(6, 4))
    expected_value: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    risk_assessment: Mapped[dict] = mapped_column(JSONB, default=dict)
    context: Mapped[dict] = mapped_column(JSONB, default=dict)
    agent_chain: Mapped[list] = mapped_column(JSONB, default=list)
    human_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    human_approver: Mapped[Optional[str]] = mapped_column(String(256))
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_decisions_ts", "timestamp"),
        Index("idx_decisions_sec", "security_id"),
        Index("idx_decisions_type", "decision_type"),
        Index("idx_decisions_approved", "approved"),
        Index("idx_decisions_hash", "decision_hash"),
    )


# ============================================================================
# 17. TRADES
# ============================================================================
class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    decision_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("decisions.id"), nullable=True)
    security_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("security_master.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[Any] = mapped_column(Numeric(20, 6), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), default="market")
    limit_price: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    order_price: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    fill_price: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    fill_quantity: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(64))
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    commission: Mapped[Any] = mapped_column(Numeric(20, 6), default=0)
    slippage: Mapped[Any] = mapped_column(Numeric(20, 6), default=0)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    decision: Mapped[Optional["Decision"]] = relationship()
    security: Mapped["SecurityMaster"] = relationship(back_populates="trades")

    __table_args__ = (
        Index("idx_trades_sec", "security_id"),
        Index("idx_trades_status", "status"),
        Index("idx_trades_paper", "is_paper"),
        Index("idx_trades_decision", "decision_id"),
    )


# ============================================================================
# 18. PORTFOLIO SNAPSHOTS
# ============================================================================
class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    positions: Mapped[list] = mapped_column(JSONB, nullable=False)
    cash: Mapped[Any] = mapped_column(Numeric(20, 6), nullable=False, default=0)
    nav: Mapped[Any] = mapped_column(Numeric(20, 6), nullable=False, default=0)
    gross_exposure: Mapped[Optional[Any]] = mapped_column(Numeric(8, 4))
    net_exposure: Mapped[Optional[Any]] = mapped_column(Numeric(8, 4))
    beta: Mapped[Optional[Any]] = mapped_column(Numeric(8, 4))
    daily_pnl: Mapped[Optional[Any]] = mapped_column(Numeric(20, 6))
    daily_pnl_pct: Mapped[Optional[Any]] = mapped_column(Numeric(8, 4))
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("as_of_date", "is_paper", name="portfolio_snapshots_key"),
        Index("idx_portfolio_date", "as_of_date"),
    )


# ============================================================================
# 19. RISK METRICS
# ============================================================================
class RiskMetric(Base):
    __tablename__ = "risk_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[Any] = mapped_column(Numeric(20, 8), nullable=False)
    metric_threshold: Mapped[Optional[Any]] = mapped_column(Numeric(20, 8))
    status: Mapped[str] = mapped_column(String(16), default="ok")
    context: Mapped[dict] = mapped_column(JSONB, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("as_of_date", "metric_name", name="risk_metrics_key"),
        Index("idx_risk_date", "as_of_date"),
        Index("idx_risk_status", "status"),
    )


# ============================================================================
# 20. DATA QUALITY LOG
# ============================================================================
class DataQualityLog(Base):
    __tablename__ = "data_quality_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    check_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    table_name: Mapped[str] = mapped_column(String(64), nullable=False)
    check_name: Mapped[str] = mapped_column(String(128), nullable=False)
    check_scope: Mapped[Optional[str]] = mapped_column(String(128))
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    severity: Mapped[str] = mapped_column(String(16), default="info")

    __table_args__ = (
        Index("idx_dq_date", "check_date"),
        Index("idx_dq_table", "table_name"),
        Index("idx_dq_sev", "severity"),
    )


# ============================================================================
# 21. INGESTION LOG
# ============================================================================
class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[Any] = mapped_column(UUID(as_uuid=True), nullable=False, server_default=func.gen_random_uuid())
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    data_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[Optional[str]] = mapped_column(String(256))
    records_written: Mapped[int] = mapped_column(BigInteger, default=0)
    records_skipped: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_ingest_provider", "provider", "data_type"),
        Index("idx_ingest_status", "status"),
        Index("idx_ingest_run", "run_id"),
    )


# ============================================================================
# MCEI CONFIG (from mcei_series_map.sql)
# ============================================================================
class MceiConfig(Base):
    __tablename__ = "mcei_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    component_name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    fred_series: Mapped[list[str]] = mapped_column(ARRAY(String(64)), nullable=False)
    sign: Mapped[str] = mapped_column(String(4), nullable=False, default="pos")
    transform: Mapped[str] = mapped_column(String(32), default="pct_yoy")
    weight: Mapped[Any] = mapped_column(Numeric(6, 4), default=0.0)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
