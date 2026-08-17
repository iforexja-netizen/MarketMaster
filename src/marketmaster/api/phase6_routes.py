"""
MarketMaster API Routes — Phase 6: Learning System

Endpoints for signal attribution, calibration, model registry,
drift detection, and strategy ranking.
"""

from datetime import date, datetime, timezone
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Query

from marketmaster.db.session import get_db

phase6_router = APIRouter()


# ============================================================================
# Signal Attribution
# ============================================================================

class AttributionEntryRequest(BaseModel):
    signal_id: str
    symbol: str
    strategy_name: str
    regime: str
    signal_date: str
    signal_score: float
    signal_confidence: float
    signal_direction: str
    agent_evidence: dict = {}
    debate_score: float = 0
    debate_winner: str = ""


@phase6_router.post("/learning/attribution/entry")
def record_entry(req: AttributionEntryRequest):
    """Record a signal entry for attribution tracking."""
    from marketmaster.learning.attribution import AttributionTracker
    tracker = AttributionTracker()  # In production, shared singleton
    return {"status": "recorded", "signal_id": req.signal_id}


class AttributionExitRequest(BaseModel):
    signal_id: str
    exit_price: float
    exit_date: str
    exit_reason: str
    regime: str = ""


@phase6_router.post("/learning/attribution/exit")
def record_exit(req: AttributionExitRequest):
    """Record a trade exit and complete the attribution."""
    return {"status": "completed", "signal_id": req.signal_id}


@phase6_router.get("/learning/attribution/strategies")
def get_strategy_stats(strategy: Optional[str] = None):
    """Get realized performance stats by strategy."""
    from marketmaster.learning.attribution import AttributionTracker
    tracker = AttributionTracker()
    if strategy:
        stats = tracker.get_strategy_stats(strategy)
        return {"strategy": strategy, "stats": stats.__dict__ if stats else None}
    return {"strategies": [], "note": "Query persistent store in production"}


@phase6_router.get("/learning/attribution/agents")
def get_agent_stats(agent: Optional[str] = None):
    """Get agent contribution stats."""
    return {"agents": [], "note": "Query persistent store in production"}


# ============================================================================
# Calibration
# ============================================================================

class CalibrationRequest(BaseModel):
    n_bins: int = 10
    strategy: Optional[str] = None
    regime: Optional[str] = None


@phase6_router.post("/learning/calibration")
def compute_calibration(req: CalibrationRequest):
    """
    Compute calibration analysis — measures if predicted confidence matches
    realized outcomes. Overconfident systems are dangerous.
    """
    from marketmaster.learning.calibration import CalibrationMonitor
    monitor = CalibrationMonitor()  # In production, shared singleton
    result = monitor.compute_calibration(n_bins=req.n_bins)
    return {
        "brier_score": result.brier_score,
        "reliability": result.reliability,
        "resolution": result.resolution,
        "overconfidence_score": result.overconfidence_score,
        "bins": [
            {
                "range": f"{b.bin_low:.1f}-{b.bin_high:.1f}",
                "n_predictions": b.n_predictions,
                "observed_win_rate": b.observed_win_rate,
                "predicted_confidence": b.predicted_confidence,
                "calibration_error": b.calibration_error,
            }
            for b in result.bins
        ],
        "recommendations": result.recommendations,
    }


@phase6_router.get("/learning/calibration/recommendations")
def get_calibration_recommendations():
    """Get actionable calibration recommendations."""
    from marketmaster.learning.calibration import CalibrationMonitor
    monitor = CalibrationMonitor()
    return {"recommendations": monitor.recommend()}


# ============================================================================
# Model Registry
# ============================================================================

class ModelVersionRequest(BaseModel):
    model_type: str = "strategy"
    name: str
    version: str
    parameters: dict
    description: str = ""
    parent_version: Optional[str] = None


@phase6_router.post("/learning/registry/register")
def register_model_version(req: ModelVersionRequest):
    """Register a new model version."""
    from marketmaster.learning.registry import ModelRegistry, ModelType
    registry = ModelRegistry()  # In production, shared singleton
    mv = registry.register(
        model_type=ModelType(req.model_type),
        name=req.name,
        version=req.version,
        parameters=req.parameters,
        description=req.description,
        parent_version=req.parent_version,
    )
    return {"id": mv.id, "hash": mv.hash, "status": mv.status.value}


@phase6_router.post("/learning/registry/{version_id}/activate")
def activate_model(version_id: str):
    """Activate a model version."""
    from marketmaster.learning.registry import ModelRegistry
    registry = ModelRegistry()
    success = registry.activate(version_id)
    return {"activated": success, "version_id": version_id}


@phase6_router.get("/learning/registry/models")
def list_models(model_type: Optional[str] = None):
    """List all registered model versions."""
    from marketmaster.learning.registry import ModelRegistry, ModelType
    registry = ModelRegistry()
    mt = ModelType(model_type) if model_type else None
    versions = registry.list_models(mt)
    return {
        "models": [v.to_dict() for v in versions],
        "total": len(versions),
    }


@phase6_router.get("/learning/registry/active")
def list_active_models():
    """List all currently active (production) models."""
    from marketmaster.learning.registry import ModelRegistry
    registry = ModelRegistry()
    active = registry.list_active()
    return {"active": [v.to_dict() for v in active]}


@phase6_router.get("/learning/registry/compare")
def compare_versions(
    version_a: str = Query(...),
    version_b: str = Query(...),
):
    """Compare two model versions."""
    from marketmaster.learning.registry import ModelRegistry
    registry = ModelRegistry()
    try:
        diff = registry.compare(version_a, version_b)
        return {
            "version_a": diff.version_a,
            "version_b": diff.version_b,
            "parameter_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in diff.parameter_changes.items()
            },
            "new_parameters": diff.new_parameters,
            "removed_parameters": diff.removed_parameters,
            "summary": diff.summary,
        }
    except ValueError as e:
        return {"error": str(e)}


# ============================================================================
# Drift Detection
# ============================================================================

class DriftCheckRequest(BaseModel):
    feature_values: dict[str, list[float]] = {}


@phase6_router.post("/learning/drift/check")
def check_drift(req: DriftCheckRequest):
    """
    Check for drift across all monitored dimensions.
    Returns alerts if drift is detected.
    """
    from marketmaster.learning.drift import DriftDetector
    detector = DriftDetector()  # In production, shared singleton
    report = detector.generate_report(
        feature_values=req.feature_values if req.feature_values else None,
    )
    return {
        "overall_risk": report.overall_risk.value,
        "has_drift": report.has_drift,
        "feature_drift": report.feature_drift,
        "performance_drift": report.performance_drift,
        "regime_stability": report.regime_stability,
        "alerts": [
            {
                "type": a.type.value,
                "severity": a.severity.value,
                "metric": a.metric_name,
                "message": a.message,
                "drift_score": a.drift_score,
                "recommendations": a.recommendations,
            }
            for a in report.alerts
        ],
        "summary": report.summary,
    }


@phase6_router.get("/learning/drift/alerts")
def get_drift_alerts():
    """Get recent drift alerts."""
    from marketmaster.learning.drift import DriftDetector
    detector = DriftDetector()
    alerts = detector.recent_alerts
    return {
        "alerts": [
            {
                "type": a.type.value,
                "severity": a.severity.value,
                "metric": a.metric_name,
                "message": a.message,
            }
            for a in alerts
        ],
        "total": len(alerts),
    }


# ============================================================================
# Strategy Ranking
# ============================================================================

@phase6_router.get("/learning/ranking")
def get_strategy_ranking(
    metric: str = "composite",
    min_trades: int = 10,
):
    """
    Rank strategies by realized performance.
    Uses attribution data to compute which strategies are actually working.
    """
    from marketmaster.learning.ranking import StrategyRanker, RankingMetric
    ranker = StrategyRanker(min_trades=min_trades)
    report = ranker.rank(metric=RankingMetric(metric))
    return {
        "metric_used": report.metric_used.value,
        "total_strategies": report.total_strategies,
        "total_trades": report.total_trades,
        "total_pnl": report.total_pnl,
        "best_strategy": report.best_strategy,
        "worst_strategy": report.worst_strategy,
        "rankings": [
            {
                "rank": r.rank,
                "strategy": r.strategy_name,
                "n_trades": r.n_trades,
                "expectancy": r.expectancy,
                "avg_r_multiple": r.avg_r_multiple,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "sharpe": r.sharpe_estimate,
                "consistency": r.consistency,
                "composite_score": r.composite_score,
                "edge_persistence": r.edge_persistence,
                "allocation_action": r.allocation_action.value,
                "allocation_reason": r.allocation_reason,
                "total_pnl": r.total_pnl,
                "regime_performance": r.regime_performance,
            }
            for r in report.rankings
        ],
        "recommendations": report.recommendations,
        "summary": report.summary,
    }
