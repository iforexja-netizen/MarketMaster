"""
Quant Engine Coordinator — Computes and stores all features.

The QuantEngine is the central computation hub that:
1. Reads raw data from the DataPlane (OHLCV, fundamentals, macro)
2. Computes technical and fundamental factors
3. Writes results to the features table (versioned, lookable by date)
4. Provides a unified interface for agents to query features

Every feature is:
- Point-in-time correct: only uses data available up to the computation date
- Versioned: feature_version tracks computation method changes
- Traceable: each feature stores its raw inputs
- Idempotent: re-computing for the same date/version skips existing records
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from marketmaster.data.plane import DataPlane
from marketmaster.db.models import (
    SecurityMaster,
    OhlcvDaily,
    Fundamentals,
    Features,
    MacroSeries,
)
from marketmaster.engines.technical import compute_all_technical, TechnicalResult
from marketmaster.engines.scoring import opportunity_score


@dataclass
class FeatureBatch:
    """Result of computing features for a security on a date."""
    security_id: int
    symbol: str
    as_of_date: date
    technical_features: dict[str, TechnicalResult] = field(default_factory=dict)
    fundamental_features: dict[str, any] = field(default_factory=dict)
    macro_features: dict[str, any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_features(self) -> int:
        return len(self.technical_features) + len(self.fundamental_features) + len(self.macro_features)


class QuantEngine:
    """
    Coordinates feature computation and storage.

    Usage:
        engine = QuantEngine(db)
        batch = engine.compute_features(security_id=1, as_of=date(2025, 1, 15))
        engine.store_features(batch)

    Or compute and store in one step:
        engine.compute_and_store(security_id=1, as_of=date(2025, 1, 15))
    """

    FEATURE_VERSION = "v1"

    def __init__(self, db: Session, benchmark_symbol: str = "SPY"):
        self.db = db
        self.plane = DataPlane(db)
        self.benchmark_symbol = benchmark_symbol
        self._benchmark_id: Optional[int] = None

    def _get_benchmark_id(self) -> Optional[int]:
        """Resolve benchmark security ID (cached)."""
        if self._benchmark_id is None:
            sec = self.plane.get_security_by_symbol(self.benchmark_symbol)
            if sec:
                self._benchmark_id = sec.id
        return self._benchmark_id

    # ========================================================================
    # Technical Features
    # ========================================================================

    def compute_technical(
        self,
        security_id: int,
        as_of: date,
        lookback_days: int = 250,  # ~1 year of trading days
    ) -> dict[str, TechnicalResult]:
        """
        Compute technical indicators for a security as of a given date.

        Point-in-time: only uses OHLCV data up to and including as_of.
        """
        start = date(as_of.year - 1, as_of.month, as_of.day)
        bars = self.plane.get_ohlcv_daily(security_id, start_date=start, end_date=as_of)

        if len(bars) < 30:
            return {}

        # Convert to pandas Series
        dates = [b.date for b in bars]
        highs = pd.Series([float(b.high) if b.high else np.nan for b in bars], index=dates)
        lows = pd.Series([float(b.low) if b.low else np.nan for b in bars], index=dates)
        closes = pd.Series([float(b.close) if b.close else np.nan for b in bars], index=dates)
        volumes = pd.Series([float(b.volume) if b.volume else 0 for b in bars], index=dates)

        # Benchmark closes
        benchmark_closes = None
        bench_id = self._get_benchmark_id()
        if bench_id and bench_id != security_id:
            bench_bars = self.plane.get_ohlcv_daily(bench_id, start_date=start, end_date=as_of)
            if bench_bars:
                bench_dates = [b.date for b in bench_bars]
                benchmark_closes = pd.Series(
                    [float(b.close) if b.close else np.nan for b in bench_bars],
                    index=bench_dates,
                )

        return compute_all_technical(highs, lows, closes, volumes, benchmark_closes)

    # ========================================================================
    # Fundamental Features
    # ========================================================================

    def compute_fundamental(
        self,
        security_id: int,
        as_of: date,
    ) -> dict[str, any]:
        """
        Compute fundamental factors for a security as of a given date.

        Point-in-time: uses fundamentals whose filing_date <= as_of.
        """
        fundamentals = self.plane.get_fundamentals(
            security_id=security_id,
            realtime_date=as_of,
        )

        if not fundamentals:
            return {}

        # Get the latest price as of the fundamental's filing date
        latest_fund = fundamentals[0]
        items = latest_fund.items if hasattr(latest_fund, 'items') else {}

        # Get latest price for ratio computation
        price = self.plane.get_latest_price(security_id, as_of)
        close = float(price.close) if price and price.close else None

        factors = {}

        # Valuation
        revenue = items.get("Revenues")
        net_income = items.get("NetIncomeLoss")
        assets = items.get("Assets")
        liabilities = items.get("Liabilities")
        equity = items.get("StockholdersEquity")
        eps = items.get("EarningsPerShareBasic")
        shares = items.get("CommonStockSharesOutstanding")
        cash = items.get("CashAndCashEquivalentsAtCarryingValue")
        operating_income = items.get("OperatingIncomeLoss")
        lt_debt = items.get("LongTermDebt")
        inventory = items.get("InventoryNet")
        ar = items.get("AccountsReceivableNetCurrent")

        # P/E
        if close and eps and eps > 0:
            factors["pe_ratio"] = close / eps

        # P/B
        if close and shares and equity and equity > 0:
            market_cap = close * shares
            factors["pb_ratio"] = market_cap / equity

        # P/S
        if close and shares and revenue and revenue > 0:
            market_cap = close * shares
            factors["ps_ratio"] = market_cap / revenue

        # ROE
        if net_income is not None and equity and equity > 0:
            factors["roe"] = net_income / equity

        # ROA
        if net_income is not None and assets and assets > 0:
            factors["roa"] = net_income / assets

        # Margins
        if revenue and revenue > 0:
            if operating_income is not None:
                factors["operating_margin"] = operating_income / revenue
            if net_income is not None:
                factors["net_margin"] = net_income / revenue
            if inventory is not None and revenue > 0:
                gross_profit = revenue - inventory  # rough approximation
                factors["gross_margin"] = gross_profit / revenue

        # Leverage
        if lt_debt is not None and equity and equity > 0:
            factors["debt_to_equity"] = lt_debt / equity
        if liabilities is not None and assets and assets > 0:
            factors["debt_to_asset"] = liabilities / assets

        # Liquidity
        if cash is not None and liabilities and liabilities > 0:
            factors["current_ratio"] = cash / liabilities  # rough — would need current liabilities

        # Growth (if we have multiple periods)
        if len(fundamentals) >= 2:
            prev_fund = fundamentals[1]
            prev_items = prev_fund.items if hasattr(prev_fund, 'items') else {}

            prev_revenue = prev_items.get("Revenues")
            prev_net_income = prev_items.get("NetIncomeLoss")
            prev_equity = prev_items.get("StockholdersEquity")
            prev_eps = prev_items.get("EarningsPerShareBasic")

            if revenue and prev_revenue and prev_revenue > 0:
                factors["revenue_growth_yoy"] = (revenue / prev_revenue) - 1
            if net_income is not None and prev_net_income and prev_net_income > 0:
                factors["earnings_growth_yoy"] = (net_income / prev_net_income) - 1
            if eps and prev_eps and prev_eps > 0:
                factors["eps_growth_yoy"] = (eps / prev_eps) - 1
            if equity and prev_equity and prev_equity > 0:
                factors["book_value_growth_yoy"] = (equity / prev_equity) - 1

        return factors

    # ========================================================================
    # Macro Features
    # ========================================================================

    def compute_macro_features(
        self,
        as_of: date,
    ) -> dict[str, any]:
        """
        Compute macro-derived features (MCEI score, regime, etc.)

        These are security-independent — the same for all securities on a given date.
        """
        features = {}

        mcei = self.plane.get_latest_mcei()
        if mcei and mcei.as_of_date <= as_of:
            features["mcei_score"] = float(mcei.score)
            features["mcei_regime"] = mcei.regime
            features["mcei_as_of_date"] = mcei.as_of_date.isoformat()

        regime = self.plane.get_latest_regime()
        if regime and regime.as_of_date <= as_of:
            features["regime"] = regime.regime
            features["regime_confidence"] = float(regime.confidence) if regime.confidence else None

        return features

    # ========================================================================
    # Full Feature Computation
    # ========================================================================

    def compute_features(
        self,
        security_id: int,
        as_of: date,
    ) -> FeatureBatch:
        """
        Compute all features (technical + fundamental + macro) for a security.

        Point-in-time: all data is sliced at as_of.
        """
        sec = self.plane.get_security_by_id(security_id)
        symbol = sec.symbol if sec else "UNKNOWN"

        batch = FeatureBatch(
            security_id=security_id,
            symbol=symbol,
            as_of_date=as_of,
        )

        # Technical
        batch.technical_features = self.compute_technical(security_id, as_of)

        # Fundamental
        batch.fundamental_features = self.compute_fundamental(security_id, as_of)

        # Macro (shared across all securities)
        batch.macro_features = self.compute_macro_features(as_of)

        return batch

    # ========================================================================
    # Storage
    # ========================================================================

    def store_features(self, batch: FeatureBatch) -> int:
        """
        Store computed features to the features table.

        Idempotent: skips features that already exist for this security/date/version.
        Returns number of features written.
        """
        written = 0

        # Technical features
        for name, result in batch.technical_features.items():
            if result.value is None:
                continue
            if self._feature_exists(batch.security_id, batch.as_of_date, name):
                continue

            feature = Features(
                security_id=batch.security_id,
                as_of_date=batch.as_of_date,
                feature_name=name,
                feature_value=float(result.value),
                feature_category="technical",
                feature_version=self.FEATURE_VERSION,
            )
            self.db.add(feature)
            written += 1

        # Fundamental features
        for name, value in batch.fundamental_features.items():
            if value is None:
                continue
            if self._feature_exists(batch.security_id, batch.as_of_date, name):
                continue

            feature = Features(
                security_id=batch.security_id,
                as_of_date=batch.as_of_date,
                feature_name=name,
                feature_value=float(value),
                feature_category="fundamental",
                feature_version=self.FEATURE_VERSION,
            )
            self.db.add(feature)
            written += 1

        # Macro features (security-independent, store with the security)
        for name, value in batch.macro_features.items():
            if value is None or isinstance(value, str):
                continue  # Skip non-numeric (regime labels)
            if self._feature_exists(batch.security_id, batch.as_of_date, name):
                continue

            feature = Features(
                security_id=batch.security_id,
                as_of_date=batch.as_of_date,
                feature_name=name,
                feature_value=float(value),
                feature_category="macro",
                feature_version=self.FEATURE_VERSION,
            )
            self.db.add(feature)
            written += 1

        self.db.commit()
        return written

    def _feature_exists(
        self,
        security_id: int,
        as_of_date: date,
        feature_name: str,
    ) -> bool:
        """Check if a feature already exists (idempotency check)."""
        existing = self.db.execute(
            select(Features).where(
                and_(
                    Features.security_id == security_id,
                    Features.as_of_date == as_of_date,
                    Features.feature_name == feature_name,
                    Features.feature_version == self.FEATURE_VERSION,
                )
            )
        ).scalars().first()
        return existing is not None

    def compute_and_store(
        self,
        security_id: int,
        as_of: date,
    ) -> tuple[FeatureBatch, int]:
        """Compute and store features in one step. Returns (batch, features_written)."""
        batch = self.compute_features(security_id, as_of)
        written = self.store_features(batch)
        return batch, written

    # ========================================================================
    # Batch Computation
    # ========================================================================

    def compute_for_universe(
        self,
        as_of: date,
        asset_class: str = "equity",
    ) -> dict[int, int]:
        """
        Compute features for all securities in the universe.

        Returns {security_id: features_written}.
        """
        securities = self.plane.get_security_master(asset_class=asset_class)
        results = {}

        for sec in securities:
            try:
                batch, written = self.compute_and_store(sec.id, as_of)
                results[sec.id] = written
            except Exception as e:
                print(f"[QuantEngine] Error computing features for {sec.symbol}: {e}")
                results[sec.id] = 0

        return results

    # ========================================================================
    # Feature Retrieval
    # ========================================================================

    def get_feature_vector(
        self,
        security_id: int,
        as_of: date,
    ) -> dict[str, float]:
        """
        Get all features for a security as of a date, as a flat dict.

        This is what agents and scoring engines use to evaluate opportunities.
        """
        features = self.plane.get_features(
            security_id=security_id,
            as_of_date=as_of,
            feature_version=self.FEATURE_VERSION,
        )

        vector = {}
        for f in features:
            if f.feature_value is not None:
                vector[f.feature_name] = float(f.feature_value)

        return vector

    def compute_opportunity_score(
        self,
        security_id: int,
        as_of: date,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Compute the opportunity score from stored features.

        Uses the scoring engine's weighted average of factor scores.
        """
        feature_vector = self.get_feature_vector(security_id, as_of)

        # Map features to scoring categories
        scores = {}

        # Technical structure (trend, momentum)
        technical_scores = []
        for key in ["rsi_14", "adx_14", "macd_histogram", "momentum_10", "roc_12"]:
            if key in feature_vector and feature_vector[key] is not None:
                technical_scores.append(feature_vector[key])
        if technical_scores:
            scores["technical_structure"] = np.mean(technical_scores)

        # Momentum
        if "momentum_10" in feature_vector:
            scores["momentum"] = feature_vector["momentum_10"]

        # Valuation
        if "pe_ratio" in feature_vector:
            # Lower P/E = higher score (simplified)
            pe = feature_vector["pe_ratio"]
            if pe > 0:
                scores["valuation"] = max(0, 100 - pe)  # crude normalization

        # Macro alignment
        if "mcei_score" in feature_vector:
            scores["macro_alignment"] = feature_vector["mcei_score"]

        if not scores:
            return 0.0

        return opportunity_score(scores, custom_weights)
