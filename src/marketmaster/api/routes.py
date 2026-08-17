"""
MarketMaster API Routes — Data Plane Endpoints

All data access goes through the DataPlane coordinator — no direct DB queries.
This ensures every consumer sees the same data with the same point-in-time semantics.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from marketmaster.db.session import get_db
from marketmaster.data.plane import DataPlane
from marketmaster.config import settings

router = APIRouter()


# ============================================================================
# Health & System
# ============================================================================

@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "marketmaster",
        "phase": "data_plane",
        "live_trading": settings.enable_live_trading,
        "version": "0.2.0",
    }


# ============================================================================
# Security Master
# ============================================================================

@router.get("/securities")
def list_securities(
    asset_class: Optional[str] = None,
    sector: Optional[str] = None,
    listing_status: str = "active",
    limit: int = Query(100, max=500),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List securities from the canonical security master."""
    plane = DataPlane(db)
    securities = plane.get_security_master(
        asset_class=asset_class,
        sector=sector,
        listing_status=listing_status,
    )
    total = len(securities)
    page = securities[offset : offset + limit]
    return {
        "total": total,
        "count": len(page),
        "securities": [
            {
                "id": s.id,
                "symbol": s.symbol,
                "name": s.name,
                "asset_class": s.asset_class,
                "exchange": s.exchange,
                "sector": s.sector,
                "industry": s.industry,
                "market_cap": s.market_cap,
                "listing_status": s.listing_status,
            }
            for s in page
        ],
    }


@router.get("/securities/{symbol}")
def get_security(symbol: str, db: Session = Depends(get_db)):
    """Get a single security by symbol."""
    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")
    return {
        "id": sec.id,
        "symbol": sec.symbol,
        "name": sec.name,
        "asset_class": sec.asset_class,
        "exchange": sec.exchange,
        "currency": sec.currency,
        "sector": sec.sector,
        "industry": sec.industry,
        "sub_industry": sec.sub_industry,
        "cik": sec.cik,
        "figi": sec.figi,
        "isin": sec.isin,
        "market_cap": sec.market_cap,
        "shares_outstanding": sec.shares_outstanding,
        "listing_status": sec.listing_status,
        "listing_date": sec.listing_date.isoformat() if sec.listing_date else None,
    }


# ============================================================================
# OHLCV
# ============================================================================

@router.get("/securities/{symbol}/ohlcv/daily")
def get_ohlcv_daily(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(1000, max=10000),
    db: Session = Depends(get_db),
):
    """Get daily OHLCV bars for a security."""
    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")

    bars = plane.get_ohlcv_daily(sec.id, start_date, end_date)
    bars = bars[-limit:] if len(bars) > limit else bars
    return {
        "symbol": symbol,
        "count": len(bars),
        "bars": [
            {
                "date": b.date.isoformat(),
                "open": float(b.open) if b.open else None,
                "high": float(b.high) if b.high else None,
                "low": float(b.low) if b.low else None,
                "close": float(b.close) if b.close else None,
                "volume": b.volume,
                "adjusted_close": float(b.adjusted_close) if b.adjusted_close else None,
                "vwap": float(b.vwap) if b.vwap else None,
            }
            for b in bars
        ],
    }


@router.get("/securities/{symbol}/price/latest")
def get_latest_price(symbol: str, as_of: Optional[date] = None, db: Session = Depends(get_db)):
    """Get the latest price for a security."""
    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")

    bar = plane.get_latest_price(sec.id, as_of)
    if not bar:
        raise HTTPException(status_code=404, detail="No price data available")
    return {
        "symbol": symbol,
        "date": bar.date.isoformat(),
        "close": float(bar.close) if bar.close else None,
        "adjusted_close": float(bar.adjusted_close) if bar.adjusted_close else None,
        "volume": bar.volume,
    }


# ============================================================================
# Macro Series
# ============================================================================

@router.get("/macro/{series_code}")
def get_macro_series(
    series_code: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    realtime_date: Optional[date] = None,
    limit: int = Query(5000, max=50000),
    db: Session = Depends(get_db),
):
    """
    Get macro series data from FRED/ALFRED.

    If realtime_date is provided, returns only data that was available
    as of that date (point-in-time query for backtesting without look-ahead).
    """
    plane = DataPlane(db)
    observations = plane.get_macro_series(series_code, start_date, end_date, realtime_date)
    observations = observations[-limit:] if len(observations) > limit else observations
    return {
        "series_code": series_code,
        "realtime_date": realtime_date.isoformat() if realtime_date else None,
        "count": len(observations),
        "observations": [
            {
                "date": o.observation_date.isoformat(),
                "value": float(o.value) if o.value else None,
                "realtime_start": o.realtime_start.isoformat() if o.realtime_start else None,
                "realtime_end": o.realtime_end.isoformat() if o.realtime_end else None,
            }
            for o in observations
        ],
    }


# ============================================================================
# MCEI
# ============================================================================

@router.get("/mcei/latest")
def get_latest_mcei(db: Session = Depends(get_db)):
    """Get the latest MCEI score and regime."""
    plane = DataPlane(db)
    mcei = plane.get_latest_mcei()
    if not mcei:
        raise HTTPException(status_code=404, detail="No MCEI data available")
    return {
        "as_of_date": mcei.as_of_date.isoformat(),
        "score": float(mcei.score),
        "regime": mcei.regime,
        "weights_version": mcei.weights_version,
        "components": mcei.components,
    }


@router.get("/mcei/history")
def get_mcei_history(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(500, max=5000),
    db: Session = Depends(get_db),
):
    """Get MCEI history with component decomposition."""
    plane = DataPlane(db)
    history = plane.get_mcei(start_date, end_date)
    history = history[:limit]
    return {
        "count": len(history),
        "history": [
            {
                "as_of_date": h.as_of_date.isoformat(),
                "score": float(h.score),
                "regime": h.regime,
                "components": h.components,
            }
            for h in history
        ],
    }


# ============================================================================
# Regime
# ============================================================================

@router.get("/regime/latest")
def get_latest_regime(db: Session = Depends(get_db)):
    """Get the latest market regime classification."""
    plane = DataPlane(db)
    regime = plane.get_latest_regime()
    if not regime:
        raise HTTPException(status_code=404, detail="No regime data available")
    return {
        "as_of_date": regime.as_of_date.isoformat(),
        "regime": regime.regime,
        "prev_regime": regime.prev_regime,
        "transition_date": regime.transition_date.isoformat() if regime.transition_date else None,
        "confidence": float(regime.confidence) if regime.confidence else None,
        "evidence": regime.evidence,
    }


@router.get("/regime/history")
def get_regime_history(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(500, max=5000),
    db: Session = Depends(get_db),
):
    """Get regime history with transitions."""
    plane = DataPlane(db)
    history = plane.get_regime_history(start_date, end_date)
    history = history[:limit]
    return {
        "count": len(history),
        "history": [
            {
                "as_of_date": r.as_of_date.isoformat(),
                "regime": r.regime,
                "prev_regime": r.prev_regime,
                "confidence": float(r.confidence) if r.confidence else None,
            }
            for r in history
        ],
    }


# ============================================================================
# Decisions (Immutable Log — Read Only)
# ============================================================================

@router.get("/decisions")
def list_decisions(
    symbol: Optional[str] = None,
    decision_type: Optional[str] = None,
    approved: Optional[bool] = None,
    limit: int = Query(50, max=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    List decisions from the immutable decision log.
    This is read-only — decisions can only be appended via log_decision().
    """
    plane = DataPlane(db)

    security_id = None
    if symbol:
        sec = plane.get_security_by_symbol(symbol)
        if not sec:
            raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")
        security_id = sec.id

    decisions = plane.get_decisions(
        security_id=security_id,
        decision_type=decision_type,
        approved=approved,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(decisions),
        "decisions": [
            {
                "id": d.id,
                "timestamp": d.timestamp.isoformat(),
                "symbol": d.symbol,
                "decision_type": d.decision_type,
                "strategy": d.strategy,
                "regime": d.regime,
                "approved": d.approved,
                "score": float(d.score) if d.score else None,
                "human_approved": d.human_approved,
                "decision_hash": d.decision_hash[:16] + "...",
                "evidence": d.evidence,
            }
            for d in decisions
        ],
    }


@router.get("/decisions/{decision_id}")
def get_decision(decision_id: int, db: Session = Depends(get_db)):
    """Get a single decision with full evidence."""
    plane = DataPlane(db)
    d = plane.get_decision(decision_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Decision not found: {decision_id}")
    return {
        "id": d.id,
        "timestamp": d.timestamp.isoformat(),
        "security_id": d.security_id,
        "symbol": d.symbol,
        "decision_type": d.decision_type,
        "strategy": d.strategy,
        "regime": d.regime,
        "approved": d.approved,
        "score": float(d.score) if d.score else None,
        "expected_value": float(d.expected_value) if d.expected_value else None,
        "evidence": d.evidence,
        "risk_assessment": d.risk_assessment,
        "context": d.context,
        "agent_chain": d.agent_chain,
        "human_approved": d.human_approved,
        "decision_hash": d.decision_hash,
        "prev_hash": d.prev_hash,
    }


# ============================================================================
# Signals
# ============================================================================

@router.get("/signals")
def list_signals(
    symbol: Optional[str] = None,
    signal_source: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(50, max=200),
    db: Session = Depends(get_db),
):
    """List trading signals with evidence."""
    plane = DataPlane(db)

    security_id = None
    if symbol:
        sec = plane.get_security_by_symbol(symbol)
        if not sec:
            raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")
        security_id = sec.id

    signals = plane.get_signals(
        security_id=security_id,
        signal_source=signal_source,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    return {
        "count": len(signals),
        "signals": [
            {
                "id": s.id,
                "symbol": symbol,
                "as_of_date": s.as_of_date.isoformat(),
                "signal_type": s.signal_type,
                "signal_source": s.signal_source,
                "direction": s.direction,
                "strength": float(s.strength) if s.strength else None,
                "score": float(s.score) if s.score else None,
                "confidence": float(s.confidence) if s.confidence else None,
                "strategy": s.strategy,
                "regime": s.regime,
            }
            for s in signals
        ],
    }


# ============================================================================
# Portfolio
# ============================================================================

@router.get("/portfolio/latest")
def get_latest_portfolio(db: Session = Depends(get_db)):
    """Get the latest portfolio snapshot."""
    plane = DataPlane(db)
    snap = plane.get_latest_portfolio()
    if not snap:
        raise HTTPException(status_code=404, detail="No portfolio data available")
    return {
        "as_of_date": snap.as_of_date.isoformat(),
        "positions": snap.positions,
        "cash": float(snap.cash),
        "nav": float(snap.nav),
        "gross_exposure": float(snap.gross_exposure) if snap.gross_exposure else None,
        "net_exposure": float(snap.net_exposure) if snap.net_exposure else None,
        "beta": float(snap.beta) if snap.beta else None,
        "daily_pnl": float(snap.daily_pnl) if snap.daily_pnl else None,
        "daily_pnl_pct": float(snap.daily_pnl_pct) if snap.daily_pnl_pct else None,
        "is_paper": snap.is_paper,
    }


# ============================================================================
# Risk Metrics
# ============================================================================

@router.get("/risk/latest")
def get_latest_risk_metrics(db: Session = Depends(get_db)):
    """Get the latest risk metrics."""
    plane = DataPlane(db)
    metrics = plane.get_latest_risk_metrics()
    return {
        "count": len(metrics),
        "metrics": [
            {
                "as_of_date": m.as_of_date.isoformat(),
                "metric_name": m.metric_name,
                "metric_value": float(m.metric_value),
                "metric_threshold": float(m.metric_threshold) if m.metric_threshold else None,
                "status": m.status,
                "context": m.context,
            }
            for m in metrics
        ],
    }


# ============================================================================
# Data Quality
# ============================================================================

@router.get("/data-quality")
def get_data_quality_summary(
    limit: int = Query(50, max=500),
    db: Session = Depends(get_db),
):
    """Get recent data quality check results."""
    from sqlalchemy import select, desc
    from marketmaster.db.models import DataQualityLog

    stmt = (
        select(DataQualityLog)
        .order_by(desc(DataQualityLog.check_date))
        .limit(limit)
    )
    results = list(db.execute(stmt).scalars().all())
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    return {
        "total_checks": len(results),
        "passed": passed,
        "failed": failed,
        "checks": [
            {
                "check_date": r.check_date.isoformat(),
                "table_name": r.table_name,
                "check_name": r.check_name,
                "passed": r.passed,
                "severity": r.severity,
                "details": r.details,
            }
            for r in results
        ],
    }


# ============================================================================
# Ingestion Status
# ============================================================================

@router.get("/ingestion/recent")
def get_recent_ingestion(
    limit: int = Query(20, max=100),
    db: Session = Depends(get_db),
):
    """Get recent ingestion run status."""
    from sqlalchemy import select, desc
    from marketmaster.db.models import IngestionLog

    stmt = (
        select(IngestionLog)
        .order_by(desc(IngestionLog.started_at))
        .limit(limit)
    )
    results = list(db.execute(stmt).scalars().all())
    return {
        "count": len(results),
        "runs": [
            {
                "run_id": str(r.run_id),
                "provider": r.provider,
                "data_type": r.data_type,
                "scope": r.scope,
                "records_written": r.records_written,
                "records_skipped": r.records_skipped,
                "status": r.status,
                "error_message": r.error_message,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in results
        ],
    }
