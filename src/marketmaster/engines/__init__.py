"""
MarketMaster engines package.

Phase 2 additions: MCEI pipeline, technical indicators, fundamental factors,
quant engine coordinator, threshold optimizer, and feature registry.
"""

from marketmaster.engines.mcei import calculate_mcei, calculate_mcei_legacy, MCEIResult
from marketmaster.engines.mcei_pipeline import MceiPipeline
from marketmaster.engines.scoring import opportunity_score, DEFAULT_WEIGHTS
from marketmaster.engines.technical import compute_all_technical, TechnicalResult
from marketmaster.engines.threshold_optimizer import ThresholdOptimizer, OptimizerReport

# Fundamental factors and feature registry may be imported
# from the sub-agent's output — try/except for safety
try:
    from marketmaster.engines.fundamental import compute_all_fundamental
except ImportError:
    pass

try:
    from marketmaster.engines.feature_registry import FeatureRegistry, FeatureSpec
except ImportError:
    pass

try:
    from marketmaster.engines.quant import QuantEngine, FeatureBatch
except ImportError:
    pass

__all__ = [
    "calculate_mcei",
    "calculate_mcei_legacy",
    "MCEIResult",
    "MceiPipeline",
    "opportunity_score",
    "DEFAULT_WEIGHTS",
    "compute_all_technical",
    "TechnicalResult",
    "ThresholdOptimizer",
    "OptimizerReport",
    "QuantEngine",
    "FeatureBatch",
    "FeatureRegistry",
    "FeatureSpec",
]
