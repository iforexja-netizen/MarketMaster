# MarketMaster

AI-native market intelligence, quantitative research, portfolio/risk, and trading platform.

Core principle: Data → Features → Regime → Specialist Agents → Strategy → Risk Gate → Portfolio → Execution → Learning.

The AI provides research and reasoning. Deterministic quantitative and risk layers remain authoritative for numerical market data, calculations, limits, and execution permissions.

## Build Sequence

1. **Data Plane** → 2. MCEI → 3. Quant Engines → 4. Regime Engine → 5. AI Agents → 6. Backtester → 7. Risk Engine → 8. Paper Trading → 9. Learning System → 10. Controlled Live Trading

## Current Status

### Phase 1 — Data Plane ✅ Schema Complete
- [x] Canonical security master (equities, ETFs, options)
- [x] Historical OHLCV (daily + intraday) with point-in-time adjusted prices
- [x] Corporate actions (splits, dividends, mergers)
- [x] Fundamentals with filing_date for point-in-time correctness
- [x] SEC filings metadata from EDGAR
- [x] Macro series from FRED/ALFRED with realtime_start/end
- [x] Economic events calendar
- [x] News items with sentiment
- [x] Earnings call transcripts
- [x] Option chains with Greeks and IV
- [x] Feature store (versioned, reproducible)
- [x] Signals with full agent evidence
- [x] MCEI history with component decomposition
- [x] Regime history with transition tracking
- [x] **Immutable decision log** (append-only, hash-chained)
- [x] Trade records linked to decisions
- [x] Portfolio snapshots
- [x] Risk metrics with thresholds
- [x] Data quality logging
- [x] Ingestion log for idempotency

### Phase 0 — Foundation ✅
- [x] Repository scaffold
- [x] FastAPI app structure
- [x] MCEI engine (stub)
- [x] Risk gate (stub)
- [x] Scoring (stub)
- [x] Strategy registry (stub)
- [x] Docker compose (PostgreSQL + Redis)

## Architecture

```
Data → Features → Regime → Specialist Agents → Strategy → Risk Gate → Portfolio → Execution → Learning
```

### Layers
1. **Data**: equities, ETFs, options, fundamentals, SEC filings, macro, news/transcripts, account data
2. **Feature Store**: returns, volatility, ATR, RSI, ADX, moving averages, relative strength, valuation, profitability, leverage, liquidity, MCEI, IV/Greeks/skew/term structure
3. **Intelligence Engines**: Fundamental, Technical, Macro/MCEI, Options, Sentiment/Research, Regime
4. **Agent Layer**: CEO/Orchestrator plus specialist agents
5. **Decision**: opportunity score → strategy → expected value → risk sizing → portfolio constraints
6. **Execution**: paper broker first; live execution behind explicit flags and hard gates
7. **Learning**: every signal, decision, order, fill and outcome is attributable

### The Immutable Decision Log
Every decision the system makes is recorded in an append-only, hash-chained table. This is the system's memory. It cannot be modified or deleted — only appended. Every trade traces back to a decision. Every decision contains the full evidence from all participating agents.

### Point-in-Time Correctness
The schema supports point-in-time queries to prevent look-ahead bias in backtests:
- `macro_series` has `realtime_start`/`realtime_end` (ALFRED vintage data)
- `fundamentals` has `filing_date` (when data was actually available)
- `ohlcv_daily` has `split_coefficient` and `adjusted_close` for proper back-adjustment

## Stack
- Python 3.11-3.13, FastAPI, PostgreSQL 16, Redis 7
- Pandas/NumPy/Polars, scikit-learn
- SQLAlchemy 2.0 + Alembic (migrations)
- Next.js/React (dashboard, planned)

## Integrations
- **Alpaca** — market data + paper trading
- **FRED/ALFRED** — macro economic data (with point-in-time vintage data)
- **SEC EDGAR** — filings + XBRL structured fundamentals

Premium data adapters remain pluggable.

## Getting Started

```bash
# Start PostgreSQL + Redis
docker-compose up -d

# Install dependencies
pip install -e ".[dev]"

# Initialize database
psql -h localhost -U marketmaster -d marketmaster -f db/schema.sql
psql -h localhost -U marketmaster -d marketmaster -f db/mcei_series_map.sql

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the API
uvicorn marketmaster.main:app --reload
```

## V1 Acceptance Criteria
MarketMaster can ingest a US equity universe, calculate core features and MCEI, classify regime, rank opportunities, generate structured research, propose strategies, calculate risk, reject invalid trades, backtest without look-ahead, paper trade, explain decisions and measure signal value.
