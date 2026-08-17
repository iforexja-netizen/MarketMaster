"""
MCEI Pipeline — Orchestrates MCEI computation from raw macro data.

Reads macro series from the DataPlane, applies the MCEI engine to compute
the composite score and regime, and persists results to mcei_history.

This is the production pipeline that ties the MCEI engine to the database.
"""

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from marketmaster.config.mcei_series import (
    MCEI_COMPONENTS,
    MCEI_REGIME_THRESHOLDS,
    get_all_series_codes,
)
from marketmaster.db.models import MacroSeries, MceiHistory, MceiConfig
from marketmaster.engines.mcei import calculate_mcei, MCEIResult
from marketmaster.data.plane import DataPlane


class MceiPipeline:
    """
    Production MCEI computation pipeline.

    Flow:
    1. Read all MCEI macro series from macro_series table (via DataPlane)
    2. For each series, apply transform (pct_yoy, level, zscore, etc.)
    3. Normalize each component (percentile or z-score)
    4. Apply sign alignment (bullish components positively, bearish negatively)
    5. Weight components into composite MCEI score
    6. Classify regime based on score
    7. Persist to mcei_history with full component decomposition
    """

    def __init__(self, db: Session):
        self.db = db
        self.plane = DataPlane(db)

    def _load_config(self) -> dict[str, dict]:
        """Load MCEI config from database (or fall back to Python config)."""
        try:
            stmt = select(MceiConfig).where(MceiConfig.is_active == True)
            configs = list(self.db.execute(stmt).scalars().all())
            if not configs:
                # Fall back to Python config
                return {
                    c.component_name: {
                        "fred_series": c.fred_series,
                        "sign": c.sign,
                        "transform": c.transform,
                        "weight": float(c.weight),
                        "category": c.category,
                        "display_name": c.display_name,
                    }
                    for c in MCEI_COMPONENTS
                }
            return {
                c.component_name: {
                    "fred_series": c.fred_series,
                    "sign": c.sign,
                    "transform": c.transform,
                    "weight": float(c.weight),
                    "category": c.category,
                    "display_name": c.display_name,
                }
                for c in configs
            }
        except Exception:
            return {
                c.component_name: {
                    "fred_series": c.fred_series,
                    "sign": c.sign,
                    "transform": c.transform,
                    "weight": float(c.weight),
                    "category": c.category,
                    "display_name": c.display_name,
                }
                for c in MCEI_COMPONENTS
            }

    def _fetch_macro_data(
        self,
        series_code: str,
        end_date: Optional[date] = None,
        start_date: Optional[date] = None,
        realtime_date: Optional[date] = None,
    ) -> pd.Series:
        """
        Fetch a macro series from the DataPlane and return as a pandas Series.

        Point-in-time: if realtime_date is set, returns only data available
        as of that date (ALFRED vintage data).
        """
        observations = self.plane.get_macro_series(
            series_code=series_code,
            start_date=start_date,
            end_date=end_date,
            realtime_date=realtime_date,
        )

        if not observations:
            return pd.Series(dtype=float)

        dates = [o.observation_date for o in observations]
        values = [float(o.value) if o.value is not None else np.nan for o in observations]

        return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_code)

    def _apply_transform(self, series: pd.Series, transform: str) -> pd.Series:
        """
        Apply the configured transform to a raw macro series.

        Transforms:
        - level: raw value
        - pct_yoy: percent change vs 1 year ago
        - pct_qoq: percent change vs 1 quarter ago
        - diff_yoy: difference vs 1 year ago
        - zscore_3y: z-score over trailing 3 years
        - zscore_5y: z-score over trailing 5 years
        """
        if series.empty:
            return series

        if transform == "level":
            return series

        if transform == "pct_yoy":
            # Year-over-year percent change
            shifted = series.shift(12)  # ~12 monthly observations = 1 year
            return ((series / shifted) - 1) * 100

        if transform == "pct_qoq":
            # Quarter-over-quarter percent change
            shifted = series.shift(3)
            return ((series / shifted) - 1) * 100

        if transform == "diff_yoy":
            # Year-over-year difference (for rates already in %)
            return series - series.shift(12)

        if transform == "zscore_3y":
            # Trailing 3-year z-score (36 monthly obs)
            rolling_mean = series.rolling(36, min_periods=12).mean()
            rolling_std = series.rolling(36, min_periods=12).std()
            return (series - rolling_mean) / rolling_std.replace(0, np.nan)

        if transform == "zscore_5y":
            # Trailing 5-year z-score (60 monthly obs)
            rolling_mean = series.rolling(60, min_periods=12).mean()
            rolling_std = series.rolling(60, min_periods=12).std()
            return (series - rolling_mean) / rolling_std.replace(0, np.nan)

        # Default: level
        return series

    def _normalize_component(self, series: pd.Series) -> pd.Series:
        """
        Normalize a transformed series to 0-100 scale using percentile rank.

        This converts any indicator to a 0-100 scale where:
        - 100 = strongest bullish reading in history
        - 0 = strongest bearish reading in history
        - 50 = median
        """
        if series.empty:
            return series

        # Use rolling percentile rank over all available history
        # This is the percentile of the current value relative to all past values
        def percentile_rank(s: pd.Series) -> float:
            clean = s.dropna()
            if len(clean) < 2:
                return 50.0
            current = clean.iloc[-1]
            rank = (clean <= current).sum() / len(clean) * 100
            return float(rank)

        # Compute expanding percentile rank
        result = pd.Series(index=series.index, dtype=float)
        clean_series = series.dropna()

        for i in range(len(clean_series)):
            window = clean_series.iloc[:i + 1]
            if len(window) < 3:
                result.loc[clean_series.index[i]] = 50.0
            else:
                current = window.iloc[-1]
                rank = (window <= current).sum() / len(window) * 100
                result.loc[clean_series.index[i]] = float(rank)

        return result

    def _apply_sign(self, series: pd.Series, sign: str) -> pd.Series:
        """
        Apply sign alignment.

        For bullish indicators (sign='pos'): higher = bullish → keep as-is
        For bearish indicators (sign='neg'): higher = bearish → invert (100 - value)
        """
        if sign == "neg":
            return 100 - series
        return series

    def compute_mcei_for_date(
        self,
        as_of: date,
        realtime_date: Optional[date] = None,
        lookback_years: int = 20,
    ) -> Optional[MCEIResult]:
        """
        Compute MCEI for a specific date using point-in-time data.

        Args:
            as_of: The date to compute MCEI for
            realtime_date: If set, uses ALFRED vintage data available as of this date
            lookback_years: How far back to fetch data for percentile normalization

        Returns MCEIResult or None if insufficient data.
        """
        config = self._load_config()
        start = date(as_of.year - lookback_years, as_of.month, 1)

        component_data: dict[str, float] = {}
        component_details: dict[str, dict] = {}

        for component_name, cfg in config.items():
            # Fetch raw data for the primary FRED series
            series_codes = cfg["fred_series"]
            raw_values = []

            for code in series_codes:
                s = self._fetch_macro_data(
                    series_code=code,
                    start_date=start,
                    end_date=as_of,
                    realtime_date=realtime_date,
                )
                if not s.empty:
                    raw_values.append(s)

            if not raw_values:
                component_details[component_name] = {"status": "no_data"}
                continue

            # If multiple series for a component, average them
            if len(raw_values) > 1:
                combined = pd.concat(raw_values, axis=1).mean(axis=1)
            else:
                combined = raw_values[0]

            # Apply transform
            transformed = self._apply_transform(combined, cfg["transform"])

            # Get the value as of our target date
            if as_of in transformed.index or pd.Timestamp(as_of) in transformed.index:
                try:
                    idx = pd.Timestamp(as_of)
                    if idx in transformed.index:
                        value = float(transformed.loc[idx])
                    else:
                        # Find the most recent value before as_of
                        mask = transformed.index <= pd.Timestamp(as_of)
                        if mask.any():
                            value = float(transformed.loc[mask].iloc[-1])
                        else:
                            continue
                except (KeyError, IndexError):
                    continue
            else:
                # Find the most recent value before as_of
                mask = transformed.index <= pd.Timestamp(as_of)
                if not mask.any():
                    continue
                value = float(transformed.loc[mask].iloc[-1])

            if np.isnan(value):
                continue

            # Normalize to percentile
            normalized = self._normalize_component(transformed)
            mask = normalized.index <= pd.Timestamp(as_of)
            if not mask.any():
                continue
            percentile_val = float(normalized.loc[mask].iloc[-1])

            if np.isnan(percentile_val):
                percentile_val = 50.0

            # Apply sign alignment
            aligned = self._apply_sign(pd.Series([percentile_val]), cfg["sign"]).iloc[0]

            component_data[component_name] = float(aligned)
            component_details[component_name] = {
                "raw_value": value,
                "percentile": percentile_val,
                "aligned": float(aligned),
                "sign": cfg["sign"],
                "transform": cfg["transform"],
                "weight": float(cfg["weight"]),
                "category": cfg.get("category"),
                "fred_series": series_codes,
            }

        if len(component_data) < 8:  # Need at least half the components
            return None

        # Compute weighted composite score
        total_weight = sum(cfg["weight"] for cfg in config.values())
        weighted_sum = sum(
            component_data.get(name, 50.0) * float(cfg["weight"])
            for name, cfg in config.items()
        )
        score = weighted_sum / total_weight if total_weight > 0 else 50.0

        # Classify regime
        regime = self._classify_regime(score)

        return MCEIResult(
            score=float(score),
            regime=regime,
            components=component_data,
            details=component_details,
            as_of_date=as_of,
        )

    def _classify_regime(self, score: float) -> str:
        """
        Classify the MCEI score into a market regime.

        Uses MCEI_REGIME_THRESHOLDS from config (dict: name -> threshold).
        Maps MCEI regimes to the broader MARKET_REGIMES where appropriate.
        """
        thresholds = MCEI_REGIME_THRESHOLDS
        # Sort by threshold descending
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)

        for regime_name, threshold_val in sorted_thresholds:
            if score >= threshold_val:
                # Map MCEI regime to market regime
                regime_map = {
                    "STRONG_EXPANSION": "STRONG_BULL",
                    "EXPANSION": "BULL",
                    "NEUTRAL": "NEUTRAL",
                    "CONTRACTION": "BEAR",
                }
                return regime_map.get(regime_name, regime_name)

        # Below the lowest threshold
        return "CRISIS"

    def compute_and_store(
        self,
        as_of: date,
        realtime_date: Optional[date] = None,
    ) -> Optional[MCEIResult]:
        """
        Compute MCEI for a date and persist to mcei_history.

        Idempotent: if a record already exists for this date, it is skipped.
        """
        # Check if already computed
        existing = self.db.execute(
            select(MceiHistory).where(MceiHistory.as_of_date == as_of).limit(1)
        ).scalars().first()

        if existing:
            return MCEIResult(
                score=float(existing.score),
                regime=existing.regime,
                components=existing.components,
                details={},
                as_of_date=as_of,
            )

        result = self.compute_mcei_for_date(as_of, realtime_date)
        if not result:
            return None

        # Persist to database
        entry = MceiHistory(
            as_of_date=as_of,
            score=result.score,
            regime=result.regime,
            components=result.components,
            weights_version="v1",
        )
        self.db.add(entry)
        self.db.commit()

        return result

    def backfill_mcei(
        self,
        start: date,
        end: date,
        realtime_date: Optional[date] = None,
        frequency: str = "monthly",
    ) -> list[MCEIResult]:
        """
        Backfill MCEI history for a date range.

        Args:
            start: Start date
            end: End date
            realtime_date: If set, uses point-in-time ALFRED data
            frequency: 'monthly' or 'weekly'

        Returns list of MCEIResult for each computed date.
        """
        results: list[MCEIResult] = []

        dates = pd.date_range(start=start, end=end, freq="MS" if frequency == "monthly" else "W")

        for dt in dates:
            as_of = dt.date()
            result = self.compute_and_store(as_of, realtime_date)
            if result:
                results.append(result)

        return results

    def get_latest_mcei(self) -> Optional[MCEIResult]:
        """Get the most recent MCEI computation from the database."""
        mcei = self.plane.get_latest_mcei()
        if not mcei:
            return None
        return MCEIResult(
            score=float(mcei.score),
            regime=mcei.regime,
            components=mcei.components,
            details={},
            as_of_date=mcei.as_of_date,
        )
