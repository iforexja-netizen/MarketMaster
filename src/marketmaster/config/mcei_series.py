"""
MCEI Component → FRED Series Mapping

This is the Python mirror of db/mcei_series_map.sql.
The ingestion layer uses this to pull the right FRED series for each MCEI component.

Signs are aligned so HIGHER = more expansionary after transformation.
Weights are initial estimates — must be validated with walk-forward testing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MCEIComponent:
    name: str
    display_name: str
    fred_series: tuple[str, ...]
    sign: str  # "pos" = higher is expansionary, "neg" = higher is contractionary
    transform: str  # pct_yoy, pct_qoq, level, spread, zscore, percentile
    weight: float
    description: str
    category: str  # money, credit, liquidity, rates, yield_curve, credit_spread, financial_conditions


# All MCEI components, ordered by category
MCEI_COMPONENTS: tuple[MCEIComponent, ...] = (
    # ── Money & Credit ──────────────────────────────────────────────────────
    MCEIComponent(
        name="broad_money_growth",
        display_name="Broad Money Growth (M2 YoY)",
        fred_series=("WM2NS",),
        sign="pos",
        transform="pct_yoy",
        weight=0.10,
        description="M2 money supply year-over-year growth rate. Higher = more expansionary monetary conditions.",
        category="money",
    ),
    MCEIComponent(
        name="bank_credit_growth",
        display_name="Bank Credit Growth (YoY)",
        fred_series=("TOTBKCR",),
        sign="pos",
        transform="pct_yoy",
        weight=0.08,
        description="Total bank credit, all commercial banks. Year-over-year growth. Expansionary when growing.",
        category="credit",
    ),
    MCEIComponent(
        name="ci_lending",
        display_name="Commercial & Industrial Loan Growth",
        fred_series=("BUSLOANS",),
        sign="pos",
        transform="pct_yoy",
        weight=0.07,
        description="C&I loans at all commercial banks. YoY growth reflects business investment appetite.",
        category="credit",
    ),
    MCEIComponent(
        name="consumer_credit",
        display_name="Consumer Credit Growth",
        fred_series=("TOTCI",),
        sign="pos",
        transform="pct_yoy",
        weight=0.05,
        description="Total consumer credit outstanding. Growth signals consumer spending capacity.",
        category="credit",
    ),

    # ── Liquidity ───────────────────────────────────────────────────────────
    MCEIComponent(
        name="fed_balance_sheet",
        display_name="Fed Balance Sheet (YoY)",
        fred_series=("WALCL",),
        sign="pos",
        transform="pct_yoy",
        weight=0.08,
        description="Federal Reserve total assets. Expansionary when growing (QE), contractionary when shrinking (QT).",
        category="liquidity",
    ),
    MCEIComponent(
        name="treasury_liquidity",
        display_name="Treasury General Account / TGA",
        fred_series=("WTREGEN",),
        sign="neg",
        transform="pct_yoy",
        weight=0.04,
        description="Treasury General Account balance. Drawdowns inject liquidity into the system.",
        category="liquidity",
    ),
    MCEIComponent(
        name="rrp_usage",
        display_name="Reverse Repo Facility Usage",
        fred_series=("RRPONTSYD",),
        sign="neg",
        transform="level",
        weight=0.03,
        description="ON RRP facility usage. High usage drains liquidity; declining usage is expansionary.",
        category="liquidity",
    ),

    # ── Rates ───────────────────────────────────────────────────────────────
    MCEIComponent(
        name="fed_funds_rate",
        display_name="Federal Funds Rate",
        fred_series=("DFF",),
        sign="neg",
        transform="level",
        weight=0.07,
        description="Effective federal funds rate. Lower rates = more expansionary.",
        category="rates",
    ),
    MCEIComponent(
        name="real_rates",
        display_name="10-Year Real Yield (TIPS)",
        fred_series=("DGS10", "DFII10"),
        sign="neg",
        transform="spread",
        weight=0.06,
        description="10-year nominal minus 10-year breakeven inflation. Lower real yields = more expansionary.",
        category="rates",
    ),

    # ── Yield Curve ─────────────────────────────────────────────────────────
    MCEIComponent(
        name="yield_curve_slope",
        display_name="Yield Curve Slope (10Y-2Y)",
        fred_series=("DGS10", "DGS2"),
        sign="pos",
        transform="spread",
        weight=0.08,
        description="10-year minus 2-year Treasury yield spread. Steeper curve = expansionary expectations.",
        category="yield_curve",
    ),
    MCEIComponent(
        name="yield_curve_3m10y",
        display_name="Yield Curve (10Y-3M)",
        fred_series=("DGS10", "DGS3MO"),
        sign="pos",
        transform="spread",
        weight=0.05,
        description="10-year minus 3-month Treasury yield spread. Classic recession predictor when inverted.",
        category="yield_curve",
    ),

    # ── Credit Spreads ──────────────────────────────────────────────────────
    MCEIComponent(
        name="credit_spread_ig",
        display_name="IG Credit Spread (BAA-AAA)",
        fred_series=("BAA", "AAA"),
        sign="neg",
        transform="spread",
        weight=0.06,
        description="Moody's BAA minus AAA corporate yield spread. Widening spreads = tighter conditions.",
        category="credit_spread",
    ),
    MCEIComponent(
        name="credit_spread_hy",
        display_name="High Yield OAS",
        fred_series=("BAMLH0A0HYM2",),
        sign="neg",
        transform="level",
        weight=0.05,
        description="ICE BofA US High Yield Index option-adjusted spread. Tighter spreads = expansionary.",
        category="credit_spread",
    ),

    # ── Financial Conditions ────────────────────────────────────────────────
    MCEIComponent(
        name="financial_conditions",
        display_name="Chicago Fed National Financial Conditions Index",
        fred_series=("NFCI",),
        sign="neg",
        transform="level",
        weight=0.06,
        description="Chicago Fed NFCI. Negative = accommodative, positive = restrictive.",
        category="financial_conditions",
    ),
    MCEIComponent(
        name="financial_conditions_leveraged",
        display_name="NFCI Leverage Subindex",
        fred_series=("NFCILEVERAGE",),
        sign="neg",
        transform="level",
        weight=0.03,
        description="Leverage component of NFCI. Captures risk appetite in funding markets.",
        category="financial_conditions",
    ),
    MCEIComponent(
        name="dxy",
        display_name="US Dollar Index",
        fred_series=("DTWEXBGS",),
        sign="neg",
        transform="pct_yoy",
        weight=0.03,
        description="Trade-weighted dollar index. Weaker dollar = more expansionary global liquidity.",
        category="financial_conditions",
    ),
)


def get_all_series_codes() -> tuple[str, ...]:
    """Return all unique FRED series codes used by MCEI components."""
    codes: set[str] = set()
    for comp in MCEI_COMPONENTS:
        codes.update(comp.fred_series)
    return tuple(sorted(codes))


def get_total_weight() -> float:
    """Return the sum of all component weights."""
    return sum(c.weight for c in MCEI_COMPONENTS)


# Regime thresholds for MCEI score (0-100)
# These are INITIAL estimates — must be validated with walk-forward testing.
MCEI_REGIME_THRESHOLDS = {
    "STRONG_EXPANSION": 80.0,
    "EXPANSION": 60.0,
    "NEUTRAL": 40.0,
    "CONTRACTION": 20.0,
    # Below 20 = STRONG_CONTRACTION
}

# Market regimes (broader than MCEI regimes — includes price/market structure)
MARKET_REGIMES = (
    "STRONG_BULL",
    "BULL",
    "TRANSITION_BULL",
    "NEUTRAL",
    "TRANSITION_BEAR",
    "BEAR",
    "CRISIS",
    "RECOVERY",
)
