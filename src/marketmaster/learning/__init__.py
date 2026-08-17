"""
MarketMaster Learning Package — Phase 6

Signal attribution, calibration monitoring, model registry, drift detection,
and strategy ranking.
"""

from marketmaster.learning.attribution import (
    SignalAttribution, AttributionTracker,
    StrategyPerformanceStats, AgentPerformanceStats, RegimePerformanceStats,
)
from marketmaster.learning.calibration import (
    CalibrationMonitor, CalibrationResult, CalibrationBin, BrierScore,
)
from marketmaster.learning.registry import (
    ModelRegistry, ModelVersion, ModelType, ModelStatus, VersionComparison,
)
from marketmaster.learning.drift import (
    DriftDetector, DriftReport, DriftAlert, DriftType, DriftSeverity,
    FeatureBaseline,
)
from marketmaster.learning.ranking import (
    StrategyRanker, StrategyRanking, RankingReport,
    RankingMetric, AllocationAction,
)

__all__ = [
    # Attribution
    "SignalAttribution", "AttributionTracker",
    "StrategyPerformanceStats", "AgentPerformanceStats", "RegimePerformanceStats",
    # Calibration
    "CalibrationMonitor", "CalibrationResult", "CalibrationBin", "BrierScore",
    # Model Registry
    "ModelRegistry", "ModelVersion", "ModelType", "ModelStatus", "VersionComparison",
    # Drift Detection
    "DriftDetector", "DriftReport", "DriftAlert", "DriftType", "DriftSeverity",
    "FeatureBaseline",
    # Strategy Ranking
    "StrategyRanker", "StrategyRanking", "RankingReport",
    "RankingMetric", "AllocationAction",
]
