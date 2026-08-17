"""
MarketMaster API Routes — Phase 2: MCEI + Quant Engines

Extends the Phase 1 data plane endpoints with:
- MCEI computation and backfill endpoints
- Feature computation endpoints
- Threshold optimizer endpoint
- Walk-forward analysis endpoint
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from marketmaster.db.session import get_db
from marketmaster.data.plane import DataPlane
from marketmaster.config import settings

phase2_router = APIRouter()


# ============================================================================
# MCEI Pipeline
# ============================================================================

@phase2_router.post("/mcei/compute")
def compute_mcei(
    as_of: Optional[date] = None,
    realtime_date: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Compute MCEI for a specific date and store to mcei_history.

    If as_of is not provided, uses today.
    If realtime_date is provided, uses point-in-time ALFRED vintage data.
    """
    from marketmaster.engines.mcei_pipeline import MceiPipeline

    if as_of is None:
        as_of = date.today()

    pipeline = MceiPipeline(db)
    result = pipeline.compute_and_store(as_of, realtime_date)

    if not result:
        raise HTTPException(
            status_code=422,
            detail="Insufficient macro data to compute MCEI. Run FRED backfill first.",
        )

    return {
        "as_of_date": result.as_of_date.isoformat(),
        "score": result.score,
        "regime": result.regime,
        "components": result.components,
        "component_details": result.details if result.details else {},
    }


@phase2_router.post("/mcei/backfill")
def backfill_mcei(
    start_date: date,
    end_date: Optional[date] = None,
    realtime_date: Optional[date] = None,
    frequency: str = "monthly",
    db: Session = Depends(get_db),
):
    """
    Backfill MCEI history for a date range.

    This computes MCEI for each date in the range and stores to mcei_history.
    Requires FRED macro data to be already ingested.
    """
    from marketmaster.engines.mcei_pipeline import MceiPipeline

    if end_date is None:
        end_date = date.today()

    pipeline = MceiPipeline(db)
    results = pipeline.backfill_mcei(start_date, end_date, realtime_date, frequency)

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "computations": len(results),
        "summary": [
            {
                "as_of_date": r.as_of_date.isoformat() if r.as_of_date else None,
                "score": r.score,
                "regime": r.regime,
            }
            for r in results
        ],
    }


# ============================================================================
# Quant Engine — Feature Computation
# ============================================================================

@phase2_router.post("/features/compute/{symbol}")
def compute_features(
    symbol: str,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Compute all features (technical + fundamental + macro) for a security
    and store to the features table.
    """
    from marketmaster.engines.quant import QuantEngine

    if as_of is None:
        as_of = date.today()

    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")

    engine = QuantEngine(db)
    batch, written = engine.compute_and_store(sec.id, as_of)

    return {
        "symbol": symbol,
        "as_of_date": as_of.isoformat(),
        "features_written": written,
        "technical_features": {
            k: {"value": v.value, "signal": v.signal}
            for k, v in batch.technical_features.items()
            if v.value is not None
        },
        "fundamental_features": batch.fundamental_features,
        "macro_features": batch.macro_features,
    }


@phase2_router.get("/features/{symbol}")
def get_features(
    symbol: str,
    as_of: Optional[date] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get stored features for a security.

    If as_of is not provided, returns the latest features.
    If category is provided, filters to that category (technical, fundamental, macro).
    """
    from marketmaster.engines.quant import QuantEngine

    if as_of is None:
        as_of = date.today()

    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")

    engine = QuantEngine(db)
    vector = engine.get_feature_vector(sec.id, as_of)

    if category:
        # Filter by category prefix
        prefixes = {
            "technical": ["rsi_", "adx_", "atr_", "sma_", "ema_", "macd_", "bollinger_",
                         "momentum_", "roc_", "stoch_", "cci_", "williams_", "obv",
                         "volume_", "relative_strength_"],
            "fundamental": ["pe_", "pb_", "ps_", "roe", "roa", "_margin", "debt_",
                           "current_ratio", "growth_", "accruals", "fcf_"],
            "macro": ["mcei_", "regime"],
        }
        relevant = [k for k in vector if any(k.startswith(p) for p in prefixes.get(category, []))]
        vector = {k: vector[k] for k in relevant}

    return {
        "symbol": symbol,
        "as_of_date": as_of.isoformat(),
        "feature_count": len(vector),
        "features": vector,
    }


@phase2_router.post("/features/compute-all")
def compute_all_features(
    as_of: Optional[date] = None,
    asset_class: str = "equity",
    db: Session = Depends(get_db),
):
    """
    Compute features for all securities in the universe.
    This is the batch computation endpoint — runs the full quant engine
    across every active security.
    """
    from marketmaster.engines.quant import QuantEngine

    if as_of is None:
        as_of = date.today()

    engine = QuantEngine(db)
    results = engine.compute_for_universe(as_of, asset_class)

    total_written = sum(results.values())
    return {
        "as_of_date": as_of.isoformat(),
        "securities_processed": len(results),
        "total_features_written": total_written,
        "per_security": {str(k): v for k, v in results.items()},
    }


@phase2_router.get("/features/{symbol}/opportunity-score")
def get_opportunity_score(
    symbol: str,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Get the opportunity score for a security based on stored features.
    Uses the weighted scoring engine to aggregate factors.
    """
    from marketmaster.engines.quant import QuantEngine

    if as_of is None:
        as_of = date.today()

    plane = DataPlane(db)
    sec = plane.get_security_by_symbol(symbol)
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security not found: {symbol}")

    engine = QuantEngine(db)
    score = engine.compute_opportunity_score(sec.id, as_of)
    feature_vector = engine.get_feature_vector(sec.id, as_of)

    return {
        "symbol": symbol,
        "as_of_date": as_of.isoformat(),
        "opportunity_score": float(score),
        "feature_count": len(feature_vector),
    }


# ============================================================================
# Threshold Optimizer — Walk-Forward Analysis
# ============================================================================

@phase2_router.post("/mcei/optimize-thresholds")
def optimize_thresholds(
    train_years: int = 10,
    test_years: int = 3,
    step_years: int = 1,
    forward_days: int = 63,
    benchmark: str = "SPY",
    db: Session = Depends(get_db),
):
    """
    Run walk-forward threshold optimization for MCEI regime boundaries.

    Tests whether the default thresholds (80=bull, 20=bear, etc.) produce
    regimes that meaningfully separate forward market returns.

    Returns stability analysis and a recommendation on whether the
    thresholds are validated or need adjustment.
    """
    from marketmaster.engines.threshold_optimizer import ThresholdOptimizer

    optimizer = ThresholdOptimizer(
        db=db,
        benchmark_symbol=benchmark,
    )

    report = optimizer.run_walk_forward(
        train_years=train_years,
        test_years=test_years,
        step_years=step_years,
        forward_days=forward_days,
    )

    return {
        "windows_analyzed": report.windows,
        "optimal_thresholds": [
            {"regime": name, "lower": low, "upper": high}
            for name, low, high in report.optimal_thresholds
        ],
        "threshold_stability": report.threshold_stability,
        "mean_train_quality": report.mean_train_sharpe,
        "mean_test_quality": report.mean_test_sharpe,
        "overfit_ratio": report.overfit_ratio,
        "regime_summary": report.regime_summary,
        "recommendation": report.recommendation,
    }


# ============================================================================
# FRED Backfill
# ============================================================================

@phase2_router.post("/backfill/fred")
async def backfill_fred(
    start_date: date = date(1990, 1, 1),
    end_date: Optional[date] = None,
    realtime: bool = False,
    db: Session = Depends(get_db),
):
    """
    Backfill FRED macro series data for MCEI computation.

    Fetches all FRED series needed for the MCEI from the start date through today.
    Set realtime=True to fetch ALFRED point-in-time vintage data for bias-free backtesting.
    """
    from marketmaster.data.backfill.fred_backfill import backfill_fred as _backfill

    if end_date is None:
        end_date = date.today()

    try:
        result = await _backfill(
            start_date=start_date,
            end_date=end_date,
            realtime=realtime,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backfill failed: {str(e)}")
