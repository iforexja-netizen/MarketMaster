"""
MarketMaster strategies package.

Phase 4: Full strategy library with 16 strategies, screener, and registry.
"""

from marketmaster.strategies.base import Strategy, SignalDirection, TradeSignal, StrategyConfig
from marketmaster.strategies.registry import StrategyRegistry
from marketmaster.strategies.screener import Screener, ScreeningResult
from marketmaster.strategies.strategies import (
    create_all_strategies,
    get_strategies_for_regime,
    TrendFollowingStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    EarningsMomentumStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    RSIReversalStrategy,
    ValueStrategy,
    QualityStrategy,
    GrowthStrategy,
    LowVolatilityStrategy,
    DefensiveStrategy,
    OptionsCollarStrategy,
    SectorRotationStrategy,
    MacroDrivenStrategy,
    RiskParityStrategy,
)

__all__ = [
    "Strategy", "SignalDirection", "TradeSignal", "StrategyConfig",
    "StrategyRegistry", "Screener", "ScreeningResult",
    "create_all_strategies", "get_strategies_for_regime",
    "TrendFollowingStrategy", "MomentumStrategy", "BreakoutStrategy",
    "EarningsMomentumStrategy", "MeanReversionStrategy", "PairsTradingStrategy",
    "RSIReversalStrategy", "ValueStrategy", "QualityStrategy", "GrowthStrategy",
    "LowVolatilityStrategy", "DefensiveStrategy", "OptionsCollarStrategy",
    "SectorRotationStrategy", "MacroDrivenStrategy", "RiskParityStrategy",
]
