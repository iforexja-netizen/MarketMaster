"""
Feature Registry — Central catalog of all MarketMaster features.

Defines what features exist, their categories, dependencies, and metadata.
Used by the QuantEngine to know what to compute, and by agents to know
what features are available for analysis.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureSpec:
    """Specification of a single feature."""
    name: str
    category: str  # technical, fundamental, macro, sentiment
    description: str
    lookback_days: Optional[int] = None
    dependencies: list[str] = field(default_factory=list)
    is_point_in_time: bool = True
    version: str = "v1"


class FeatureRegistry:
    """
    Registry of all available features.

    Pre-populated with all technical, fundamental, macro, and sentiment features
    defined in the MarketMaster system.
    """

    def __init__(self):
        self._features: dict[str, FeatureSpec] = {}
        self._register_defaults()

    def register(self, spec: FeatureSpec) -> None:
        """Register a new feature."""
        if spec.name in self._features:
            raise ValueError(f"Feature already registered: {spec.name}")
        self._features[spec.name] = spec

    def get(self, name: str) -> Optional[FeatureSpec]:
        """Get a feature specification by name."""
        return self._features.get(name)

    def all(self) -> list[FeatureSpec]:
        """Get all registered features."""
        return list(self._features.values())

    def by_category(self, category: str) -> list[FeatureSpec]:
        """Get all features in a category."""
        return [f for f in self._features.values() if f.category == category]

    def list_dependencies(self, name: str) -> list[str]:
        """List the dependencies of a feature."""
        spec = self.get(name)
        return spec.dependencies if spec else []

    def categories(self) -> list[str]:
        """Get all unique categories."""
        return sorted(set(f.category for f in self._features.values()))

    def count(self) -> int:
        """Total number of registered features."""
        return len(self._features)

    def _register_defaults(self) -> None:
        """Register all default MarketMaster features."""

        # ── Technical Features ───────────────────────────────────────────────
        technical_specs = [
            ("rsi_14", "Relative Strength Index (14-period)", 14, []),
            ("adx_14", "Average Directional Index (14-period)", 28, []),
            ("atr_14", "Average True Range (14-period)", 14, []),
            ("sma_20", "Simple Moving Average (20-day)", 20, []),
            ("sma_50", "Simple Moving Average (50-day)", 50, []),
            ("sma_200", "Simple Moving Average (200-day)", 200, []),
            ("ema_12", "Exponential Moving Average (12-day)", 12, []),
            ("ema_26", "Exponential Moving Average (26-day)", 26, []),
            ("macd", "MACD Line (12-26)", 26, ["ema_12", "ema_26"]),
            ("macd_signal", "MACD Signal Line (9-period)", 35, ["macd"]),
            ("macd_histogram", "MACD Histogram", 35, ["macd", "macd_signal"]),
            ("bollinger_upper", "Bollinger Band Upper (20, 2σ)", 20, []),
            ("bollinger_middle", "Bollinger Band Middle (20)", 20, []),
            ("bollinger_lower", "Bollinger Band Lower (20, 2σ)", 20, []),
            ("bollinger_width", "Bollinger Band Width", 20, ["bollinger_upper", "bollinger_lower"]),
            ("bollinger_position", "Position within Bollinger Bands", 20, ["bollinger_upper", "bollinger_lower"]),
            ("momentum_10", "Momentum (10-day rate of change)", 10, []),
            ("roc_12", "Rate of Change (12-period)", 12, []),
            ("stoch_k", "Stochastic %K (14-period)", 14, []),
            ("stoch_d", "Stochastic %D (3-period)", 17, ["stoch_k"]),
            ("cci_20", "Commodity Channel Index (20-period)", 20, []),
            ("williams_r_14", "Williams %R (14-period)", 14, []),
            ("obv", "On-Balance Volume", 1, []),
            ("volume_sma_20", "Volume SMA (20-day)", 20, []),
            ("volume_ratio", "Volume Ratio vs SMA", 21, ["volume_sma_20"]),
            ("relative_strength_60", "Relative Strength vs Benchmark (60-day)", 60, []),
        ]

        for name, desc, lookback, deps in technical_specs:
            self.register(FeatureSpec(
                name=name, category="technical", description=desc,
                lookback_days=lookback, dependencies=deps,
            ))

        # ── Fundamental Features ─────────────────────────────────────────────
        fundamental_specs = [
            ("pe_ratio", "Price-to-Earnings ratio", None, []),
            ("pb_ratio", "Price-to-Book ratio", None, []),
            ("ps_ratio", "Price-to-Sales ratio", None, []),
            ("ev_ebitda", "Enterprise Value / EBITDA", None, []),
            ("fcf_yield", "Free Cash Flow Yield", None, []),
            ("roe", "Return on Equity", None, []),
            ("roa", "Return on Assets", None, []),
            ("gross_margin", "Gross Profit Margin", None, []),
            ("operating_margin", "Operating Margin", None, []),
            ("net_margin", "Net Profit Margin", None, []),
            ("roic", "Return on Invested Capital", None, []),
            ("debt_to_equity", "Debt-to-Equity ratio", None, []),
            ("debt_to_asset", "Debt-to-Asset ratio", None, []),
            ("interest_coverage", "Interest Coverage Ratio", None, []),
            ("current_ratio", "Current Ratio", None, []),
            ("quick_ratio", "Quick Ratio", None, []),
            ("revenue_growth_yoy", "Revenue Growth Year-over-Year", None, []),
            ("earnings_growth_yoy", "Earnings Growth Year-over-Year", None, []),
            ("eps_growth_yoy", "EPS Growth Year-over-Year", None, []),
            ("book_value_growth_yoy", "Book Value Growth Year-over-Year", None, []),
            ("accruals_ratio", "Accruals Ratio (earnings quality)", None, []),
            ("fcf_to_net_income", "FCF / Net Income (cash backing)", None, []),
        ]

        for name, desc, lookback, deps in fundamental_specs:
            self.register(FeatureSpec(
                name=name, category="fundamental", description=desc,
                lookback_days=lookback, dependencies=deps,
            ))

        # ── Macro Features ──────────────────────────────────────────────────
        macro_specs = [
            ("mcei_score", "MarketMaster Composite Economic Indicator score", None, []),
            ("mcei_regime", "MCEI regime classification", None, ["mcei_score"]),
            ("mcei_momentum", "MCEI score momentum (change vs prior period)", None, ["mcei_score"]),
            ("regime", "Current market regime", None, ["mcei_regime"]),
            ("regime_confidence", "Regime classification confidence", None, ["regime"]),
        ]

        for name, desc, lookback, deps in macro_specs:
            self.register(FeatureSpec(
                name=name, category="macro", description=desc,
                lookback_days=lookback, dependencies=deps,
            ))

        # ── Sentiment Features (stubs for Phase 5) ───────────────────────────
        sentiment_specs = [
            ("sentiment_score", "News sentiment score (-1 to 1)", None, []),
            ("news_volume", "News article volume (count per day)", None, []),
            ("sentiment_momentum", "Sentiment score change vs prior period", None, ["sentiment_score"]),
            ("social_sentiment", "Social media sentiment score", None, []),
            ("analyst_consensus", "Analyst consensus rating", None, []),
            ("put_call_ratio", "Put/Call ratio from options", None, []),
        ]

        for name, desc, lookback, deps in sentiment_specs:
            self.register(FeatureSpec(
                name=name, category="sentiment", description=desc,
                lookback_days=lookback, dependencies=deps,
            ))
