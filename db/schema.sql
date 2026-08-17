-- ============================================================================
-- MarketMaster — Data Plane Schema (Phase 1)
-- ============================================================================
-- This is the canonical source of truth. Every agent reads from here.
-- No agent independently pulls data and reaches different conclusions.
--
-- Design principles:
--   1. Point-in-time correctness — no look-ahead bias in backtests
--   2. Immutable decision log — append-only, hash-chained for audit
--   3. Full attribution — every signal, decision, fill, outcome is traceable
--   4. Multi-asset — equities, ETFs, options, macro series
--   5. Feature versioning — features are reproducible and comparable
-- ============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. SECURITY MASTER — Canonical instrument universe
-- ============================================================================
CREATE TABLE security_master (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(32)   NOT NULL,
    name            VARCHAR(512),
    asset_class     VARCHAR(32)   NOT NULL DEFAULT 'equity',
    exchange        VARCHAR(32),
    currency        VARCHAR(8)    NOT NULL DEFAULT 'USD',
    sector          VARCHAR(128),
    industry        VARCHAR(128),
    sub_industry    VARCHAR(128),
    cik             VARCHAR(16),
    figi            VARCHAR(20),
    isin            VARCHAR(16),
    cusip           VARCHAR(12),
    composite_figi  VARCHAR(20),
    market_cap      BIGINT,
    shares_outstanding BIGINT,
    listing_status  VARCHAR(16)   NOT NULL DEFAULT 'active',
    listing_date    DATE,
    delisting_date  DATE,
    tick_size       DECIMAL(12,6),
    currency_primary VARCHAR(8)  DEFAULT 'USD',
    meta            JSONB         DEFAULT '{}'::jsonb,
    created_date    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_date    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, exchange)
);

CREATE INDEX idx_sec_master_symbol     ON security_master (symbol);
CREATE INDEX idx_sec_master_asset_class ON security_master (asset_class);
CREATE INDEX idx_sec_master_sector     ON security_master (sector);
CREATE INDEX idx_sec_master_cik        ON security_master (cik);
CREATE INDEX idx_sec_master_status     ON security_master (listing_status);

-- ============================================================================
-- 2. OHLCV DAILY — Historical daily bars (point-in-time)
-- ============================================================================
CREATE TABLE ohlcv_daily (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    date            DATE          NOT NULL,
    open            DECIMAL(20,6),
    high            DECIMAL(20,6),
    low             DECIMAL(20,6),
    close           DECIMAL(20,6),
    volume          BIGINT,
    adjusted_close  DECIMAL(20,6),
    dividend_amount DECIMAL(20,6) DEFAULT 0,
    split_coefficient DECIMAL(20,10) DEFAULT 1,
    vwap            DECIMAL(20,6),
    source          VARCHAR(32)   NOT NULL DEFAULT 'alpaca',
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, date)
);

CREATE INDEX idx_ohlcv_daily_sec_date ON ohlcv_daily (security_id, date DESC);
CREATE INDEX idx_ohlcv_daily_date     ON ohlcv_daily (date);

-- ============================================================================
-- 3. OHLCV INTRADAY — Intraday bars (1m, 5m, 15m, 1h)
-- ============================================================================
CREATE TABLE ohlcv_intraday (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    timestamp       TIMESTAMPTZ   NOT NULL,
    interval        VARCHAR(8)    NOT NULL,
    open            DECIMAL(20,6),
    high            DECIMAL(20,6),
    low             DECIMAL(20,6),
    close           DECIMAL(20,6),
    volume          BIGINT,
    vwap            DECIMAL(20,6),
    source          VARCHAR(32)   NOT NULL DEFAULT 'alpaca',
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, timestamp, interval)
);

CREATE INDEX idx_ohlcv_intra_sec_ts ON ohlcv_intraday (security_id, timestamp DESC);
CREATE INDEX idx_ohlcv_intra_interval ON ohlcv_intraday (interval);

-- ============================================================================
-- 4. CORPORATE ACTIONS — Splits, dividends, mergers, spinoffs
-- ============================================================================
CREATE TABLE corporate_actions (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    ex_date         DATE          NOT NULL,
    action_type     VARCHAR(32)   NOT NULL,
    description     VARCHAR(512),
    value           DECIMAL(20,6),
    value_fractional DECIMAL(20,10),
    source          VARCHAR(32)   NOT NULL DEFAULT 'alpaca',
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, ex_date, action_type)
);

CREATE INDEX idx_corp_actions_sec_date ON corporate_actions (security_id, ex_date DESC);

-- ============================================================================
-- 5. FUNDAMENTALS — Financial statement data (point-in-time)
-- ============================================================================
CREATE TABLE fundamentals (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    report_date     DATE          NOT NULL,
    fiscal_year     INT,
    fiscal_quarter  INT,
    period_type     VARCHAR(16)   NOT NULL,
    statement_type  VARCHAR(32)   NOT NULL,
    items           JSONB         NOT NULL DEFAULT '{}'::jsonb,
    source          VARCHAR(32)   NOT NULL DEFAULT 'sec_edgar',
    filing_date     DATE,
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, report_date, period_type, statement_type)
);

CREATE INDEX idx_fundamentals_sec_date ON fundamentals (security_id, report_date DESC);
CREATE INDEX idx_fundamentals_filing    ON fundamentals (filing_date);

-- ============================================================================
-- 6. SEC FILINGS — Filing metadata from EDGAR
-- ============================================================================
CREATE TABLE sec_filings (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        REFERENCES security_master(id),
    cik             VARCHAR(16)   NOT NULL,
    accession_no    VARCHAR(32)   NOT NULL,
    filing_date     DATE          NOT NULL,
    form_type       VARCHAR(16)   NOT NULL,
    description     TEXT,
    primary_document VARCHAR(256),
    filing_url      VARCHAR(512),
    parsed          BOOLEAN       DEFAULT FALSE,
    parsed_data     JSONB         DEFAULT '{}'::jsonb,
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (cik, accession_no)
);

CREATE INDEX idx_sec_filings_cik_date ON sec_filings (cik, filing_date DESC);
CREATE INDEX idx_sec_filings_form     ON sec_filings (form_type);

-- ============================================================================
-- 7. MACRO SERIES — FRED/ALFRED economic data (point-in-time)
-- ============================================================================
CREATE TABLE macro_series (
    id              BIGSERIAL PRIMARY KEY,
    series_code     VARCHAR(64)   NOT NULL,
    series_name     VARCHAR(256),
    source          VARCHAR(32)   NOT NULL DEFAULT 'fred',
    frequency       VARCHAR(16),
    units           VARCHAR(64),
    seasonally_adj  BOOLEAN      DEFAULT FALSE,
    observation_date DATE        NOT NULL,
    value           DECIMAL(24,8),
    realtime_start  DATE,
    realtime_end    DATE,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (series_code, observation_date, COALESCE(realtime_start, '1900-01-01'::date))
);

CREATE INDEX idx_macro_series_code_date ON macro_series (series_code, observation_date DESC);
CREATE INDEX idx_macro_series_realtime  ON macro_series (series_code, realtime_start, realtime_end);

-- ============================================================================
-- 8. ECONOMIC EVENTS — Economic calendar
-- ============================================================================
CREATE TABLE economic_events (
    id              BIGSERIAL PRIMARY KEY,
    event_date      DATE          NOT NULL,
    event_time      TIME,
    country         VARCHAR(8)    NOT NULL DEFAULT 'US',
    event_name      VARCHAR(256)  NOT NULL,
    importance      VARCHAR(16)   DEFAULT 'medium',
    actual          DECIMAL(24,8),
    forecast        DECIMAL(24,8),
    previous        DECIMAL(24,8),
    actual_unit     VARCHAR(32),
    source          VARCHAR(64),
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (event_date, event_name, country)
);

CREATE INDEX idx_econ_events_date ON economic_events (event_date);

-- ============================================================================
-- 9. NEWS ITEMS — News headlines and metadata
-- ============================================================================
CREATE TABLE news_items (
    id              BIGSERIAL PRIMARY KEY,
    news_date       TIMESTAMPTZ   NOT NULL,
    source          VARCHAR(64)   NOT NULL,
    headline        TEXT         NOT NULL,
    summary         TEXT,
    url             VARCHAR(512),
    symbols         VARCHAR(32)[],
    sentiment_score DECIMAL(6,4),
    sentiment_label VARCHAR(16),
    relevance_score DECIMAL(6,4),
    raw             JSONB         DEFAULT '{}'::jsonb,
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_news_date    ON news_items (news_date DESC);
CREATE INDEX idx_news_symbols ON news_items USING GIN (symbols);

-- ============================================================================
-- 10. TRANSCRIPTS — Earnings call transcripts
-- ============================================================================
CREATE TABLE transcripts (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    call_date       TIMESTAMPTZ   NOT NULL,
    fiscal_year     INT,
    fiscal_quarter  INT,
    transcript_type VARCHAR(32)   DEFAULT 'earnings_call',
    raw_text        TEXT,
    parsed          JSONB         DEFAULT '{}'::jsonb,
    source          VARCHAR(64),
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, call_date)
);

CREATE INDEX idx_transcripts_sec_date ON transcripts (security_id, call_date DESC);

-- ============================================================================
-- 11. OPTION CHAINS — Options data snapshot (Greeks, IV, OI)
-- ============================================================================
CREATE TABLE option_chains (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    as_of_date      DATE          NOT NULL,
    expiration      DATE          NOT NULL,
    strike          DECIMAL(12,4) NOT NULL,
    option_type     VARCHAR(4)    NOT NULL,
    symbol          VARCHAR(64),
    bid             DECIMAL(20,6),
    ask             DECIMAL(20,6),
    last            DECIMAL(20,6),
    volume          BIGINT,
    open_interest   BIGINT,
    iv              DECIMAL(10,6),
    delta           DECIMAL(10,6),
    gamma           DECIMAL(12,8),
    theta           DECIMAL(12,8),
    vega            DECIMAL(12,8),
    rho             DECIMAL(12,8),
    underlying_price DECIMAL(20,6),
    source          VARCHAR(32)   NOT NULL DEFAULT 'alpaca',
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, as_of_date, expiration, strike, option_type)
);

CREATE INDEX idx_options_sec_date ON option_chains (security_id, as_of_date DESC);
CREATE INDEX idx_options_expiry   ON option_chains (expiration);

-- ============================================================================
-- 12. FEATURES — Computed feature store (versioned, reproducible)
-- ============================================================================
CREATE TABLE features (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    as_of_date      DATE          NOT NULL,
    feature_name    VARCHAR(128)  NOT NULL,
    feature_value   DECIMAL(24,10),
    feature_category VARCHAR(32),
    feature_version VARCHAR(16)  DEFAULT 'v1',
    lookback_days   INT,
    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (security_id, as_of_date, feature_name, feature_version)
);

CREATE INDEX idx_features_sec_date ON features (security_id, as_of_date DESC);
CREATE INDEX idx_features_name      ON features (feature_name);
CREATE INDEX idx_features_category  ON features (feature_category);

-- ============================================================================
-- 13. SIGNALS — Agent-generated trading signals
-- ============================================================================
CREATE TABLE signals (
    id              BIGSERIAL PRIMARY KEY,
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    as_of_date      DATE          NOT NULL,
    signal_type     VARCHAR(64)   NOT NULL,
    signal_source   VARCHAR(64)   NOT NULL,
    direction       VARCHAR(8)    NOT NULL,
    strength        DECIMAL(6,4),
    score           DECIMAL(6,4),
    confidence      DECIMAL(6,4),
    strategy        VARCHAR(64),
    regime          VARCHAR(32),
    evidence        JSONB         DEFAULT '{}'::jsonb,
    data_quality    DECIMAL(6,4),
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signals_sec_date  ON signals (security_id, as_of_date DESC);
CREATE INDEX idx_signals_source    ON signals (signal_source);
CREATE INDEX idx_signals_date      ON signals (as_of_date DESC);

-- ============================================================================
-- 14. MCEI HISTORY — Macro Conditions & Expectations Index over time
-- ============================================================================
CREATE TABLE mcei_history (
    id              BIGSERIAL PRIMARY KEY,
    as_of_date      DATE          NOT NULL UNIQUE,
    score           DECIMAL(6,2)  NOT NULL,
    regime          VARCHAR(32)   NOT NULL,
    components      JSONB         NOT NULL,
    weights_version VARCHAR(16)  DEFAULT 'v1',
    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_mcei_date ON mcei_history (as_of_date DESC);

-- ============================================================================
-- 15. REGIME HISTORY — Market regime classifications over time
-- ============================================================================
CREATE TABLE regime_history (
    id              BIGSERIAL PRIMARY KEY,
    as_of_date      DATE          NOT NULL UNIQUE,
    regime          VARCHAR(32)   NOT NULL,
    prev_regime     VARCHAR(32),
    transition_date DATE,
    confidence      DECIMAL(6,4),
    evidence        JSONB         DEFAULT '{}'::jsonb,
    regime_version  VARCHAR(16)   DEFAULT 'v1',
    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_regime_date ON regime_history (as_of_date DESC);

-- ============================================================================
-- 16. DECISIONS — IMMUTABLE DECISION LOG (append-only, hash-chained)
-- ============================================================================
-- The most important table. Every decision the system makes is recorded here.
-- It cannot be updated or deleted — only appended.
-- Each decision is hash-chained to the previous one for tamper detection.
CREATE TABLE decisions (
    id              BIGSERIAL PRIMARY KEY,
    decision_hash   VARCHAR(64)   NOT NULL UNIQUE,
    prev_hash       VARCHAR(64),
    timestamp       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    security_id     BIGINT        REFERENCES security_master(id),
    symbol          VARCHAR(32),
    decision_type   VARCHAR(32)   NOT NULL,
    strategy        VARCHAR(64),
    regime          VARCHAR(32),
    approved        BOOLEAN       NOT NULL DEFAULT FALSE,
    score           DECIMAL(6,4),
    expected_value  DECIMAL(20,6),
    evidence        JSONB         NOT NULL,
    risk_assessment JSONB         DEFAULT '{}'::jsonb,
    context         JSONB         DEFAULT '{}'::jsonb,
    agent_chain     JSONB         DEFAULT '[]'::jsonb,
    human_approved  BOOLEAN       DEFAULT FALSE,
    human_approver  VARCHAR(256),
    approved_at     TIMESTAMPTZ
);

CREATE INDEX idx_decisions_ts       ON decisions (timestamp DESC);
CREATE INDEX idx_decisions_sec       ON decisions (security_id);
CREATE INDEX idx_decisions_type      ON decisions (decision_type);
CREATE INDEX idx_decisions_approved  ON decisions (approved);
CREATE INDEX idx_decisions_hash      ON decisions (decision_hash);

-- Append-only enforcement: prevent UPDATE and DELETE
CREATE OR REPLACE FUNCTION prevent_decision_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Decisions table is append-only. Modification is not permitted.';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_decisions_no_update
    BEFORE UPDATE ON decisions
    FOR EACH ROW EXECUTE FUNCTION prevent_decision_modification();

CREATE TRIGGER tr_decisions_no_delete
    BEFORE DELETE ON decisions
    FOR EACH ROW EXECUTE FUNCTION prevent_decision_modification();

-- ============================================================================
-- 17. TRADES — Paper and live trade records
-- ============================================================================
CREATE TABLE trades (
    id              BIGSERIAL PRIMARY KEY,
    decision_id     BIGINT        REFERENCES decisions(id),
    security_id     BIGINT        NOT NULL REFERENCES security_master(id),
    symbol          VARCHAR(32)   NOT NULL,
    side            VARCHAR(8)    NOT NULL,
    quantity        DECIMAL(20,6) NOT NULL,
    order_type      VARCHAR(16)   DEFAULT 'market',
    limit_price     DECIMAL(20,6),
    order_price     DECIMAL(20,6),
    fill_price      DECIMAL(20,6),
    fill_quantity   DECIMAL(20,6),
    status          VARCHAR(16)   NOT NULL DEFAULT 'pending',
    broker_order_id VARCHAR(64),
    is_paper        BOOLEAN       NOT NULL DEFAULT TRUE,
    commission      DECIMAL(20,6) DEFAULT 0,
    slippage        DECIMAL(20,6) DEFAULT 0,
    placed_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    filled_at       TIMESTAMPTZ,
    notes           TEXT,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_sec       ON trades (security_id);
CREATE INDEX idx_trades_status    ON trades (status);
CREATE INDEX idx_trades_paper     ON trades (is_paper);
CREATE INDEX idx_trades_decision  ON trades (decision_id);

-- ============================================================================
-- 18. PORTFOLIO SNAPSHOTS — Daily portfolio state
-- ============================================================================
CREATE TABLE portfolio_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    as_of_date      DATE          NOT NULL,
    positions        JSONB         NOT NULL,
    cash            DECIMAL(20,6) NOT NULL DEFAULT 0,
    nav             DECIMAL(20,6) NOT NULL DEFAULT 0,
    gross_exposure  DECIMAL(8,4),
    net_exposure    DECIMAL(8,4),
    beta            DECIMAL(8,4),
    daily_pnl       DECIMAL(20,6),
    daily_pnl_pct   DECIMAL(8,4),
    is_paper        BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (as_of_date, is_paper)
);

CREATE INDEX idx_portfolio_date ON portfolio_snapshots (as_of_date DESC);

-- ============================================================================
-- 19. RISK METRICS — Daily risk measurements
-- ============================================================================
CREATE TABLE risk_metrics (
    id              BIGSERIAL PRIMARY KEY,
    as_of_date      DATE          NOT NULL,
    metric_name     VARCHAR(128)  NOT NULL,
    metric_value    DECIMAL(20,8) NOT NULL,
    metric_threshold DECIMAL(20,8),
    status          VARCHAR(16)   DEFAULT 'ok',
    context         JSONB         DEFAULT '{}'::jsonb,
    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (as_of_date, metric_name)
);

CREATE INDEX idx_risk_date   ON risk_metrics (as_of_date DESC);
CREATE INDEX idx_risk_status ON risk_metrics (status);

-- ============================================================================
-- 20. DATA QUALITY LOG — Freshness, completeness, anomaly checks
-- ============================================================================
CREATE TABLE data_quality_log (
    id              BIGSERIAL PRIMARY KEY,
    check_date      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    table_name      VARCHAR(64)   NOT NULL,
    check_name      VARCHAR(128)  NOT NULL,
    check_scope      VARCHAR(128),
    passed           BOOLEAN      NOT NULL,
    details          JSONB        DEFAULT '{}'::jsonb,
    severity         VARCHAR(16)  DEFAULT 'info'
);

CREATE INDEX idx_dq_date  ON data_quality_log (check_date DESC);
CREATE INDEX idx_dq_table ON data_quality_log (table_name);
CREATE INDEX idx_dq_sev   ON data_quality_log (severity);

-- ============================================================================
-- 21. DATA INGESTION LOG — Track ingestion runs for idempotency
-- ============================================================================
CREATE TABLE ingestion_log (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID         NOT NULL DEFAULT uuid_generate_v4(),
    provider        VARCHAR(32)  NOT NULL,
    data_type       VARCHAR(64)  NOT NULL,
    scope           VARCHAR(256),
    records_written BIGINT       DEFAULT 0,
    records_skipped BIGINT       DEFAULT 0,
    status          VARCHAR(16)  NOT NULL,
    error_message   TEXT,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_ingest_provider ON ingestion_log (provider, data_type);
CREATE INDEX idx_ingest_status   ON ingestion_log (status);
CREATE INDEX idx_ingest_run      ON ingestion_log (run_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

CREATE OR REPLACE VIEW v_latest_ohlcv AS
SELECT DISTINCT ON (s.security_id)
    s.security_id,
    sm.symbol,
    sm.name,
    sm.sector,
    sm.industry,
    s.date,
    s.open, s.high, s.low, s.close, s.volume, s.adjusted_close
FROM ohlcv_daily s
JOIN security_master sm ON s.security_id = sm.id
ORDER BY s.security_id, s.date DESC;

CREATE OR REPLACE VIEW v_current_regime AS
SELECT
    m.as_of_date,
    m.score AS mcei_score,
    m.regime AS mcei_regime,
    m.components,
    r.regime AS market_regime,
    r.prev_regime,
    r.confidence AS regime_confidence
FROM mcei_history m
FULL OUTER JOIN regime_history r ON m.as_of_date = r.as_of_date
ORDER BY m.as_of_date DESC
LIMIT 1;

CREATE OR REPLACE VIEW v_decision_chain AS
SELECT
    d.id, d.timestamp, d.symbol, d.decision_type, d.strategy, d.regime,
    d.approved, d.score, d.human_approved,
    d.decision_hash, d.prev_hash,
    d.agent_chain
FROM decisions d
ORDER BY d.timestamp DESC
LIMIT 200;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE security_master IS 'Canonical instrument universe. One source of truth for all symbols, metadata, and listing status.';
COMMENT ON TABLE ohlcv_daily IS 'Historical daily OHLCV bars. Point-in-time: adjusted_close and split_coefficient enable backtesting without look-ahead.';
COMMENT ON TABLE ohlcv_intraday IS 'Intraday bars at various intervals for real-time and intraday strategies.';
COMMENT ON TABLE corporate_actions IS 'Splits, dividends, mergers. Essential for adjusted price calculation and point-in-time correctness.';
COMMENT ON TABLE fundamentals IS 'Financial statement data. filing_date is when data was actually available — critical for avoiding look-ahead.';
COMMENT ON TABLE sec_filings IS 'SEC EDGAR filing metadata. parsed flag tracks whether structured data has been extracted.';
COMMENT ON TABLE macro_series IS 'FRED/ALFRED macro data. realtime_start/end enable point-in-time queries for backtesting.';
COMMENT ON TABLE economic_events IS 'Economic calendar with actual/forecast/previous values for event-driven analysis.';
COMMENT ON TABLE news_items IS 'News headlines with sentiment scoring. symbols array enables cross-referencing with security_master.';
COMMENT ON TABLE transcripts IS 'Earnings call transcripts. parsed JSONB contains structured speaker/section extraction.';
COMMENT ON TABLE option_chains IS 'Options snapshots with Greeks, IV, and open interest. as_of_date + expiration for time-series analysis.';
COMMENT ON TABLE features IS 'Computed feature store. feature_version ensures reproducibility across model iterations.';
COMMENT ON TABLE signals IS 'Agent-generated trading signals with full evidence. Every signal is traceable to its source agent.';
COMMENT ON TABLE mcei_history IS 'MCEI composite score history. components JSONB enables decomposition and threshold testing.';
COMMENT ON TABLE regime_history IS 'Market regime classifications. transition tracking and evidence for explainability.';
COMMENT ON TABLE decisions IS 'IMMUTABLE decision log. Append-only with hash-chaining for tamper detection. The system memory.';
COMMENT ON TABLE trades IS 'Paper and live trade records linked to decisions. Every trade traces back to a decision.';
COMMENT ON TABLE portfolio_snapshots IS 'Daily portfolio state snapshots. positions JSONB contains full position detail.';
COMMENT ON TABLE risk_metrics IS 'Daily risk measurements with thresholds and breach status.';
COMMENT ON TABLE data_quality_log IS 'Data quality checks: freshness, completeness, null detection, range validation.';
COMMENT ON TABLE ingestion_log IS 'Ingestion run tracking for idempotency and observability.';
