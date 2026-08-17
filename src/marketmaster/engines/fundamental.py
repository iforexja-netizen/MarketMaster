"""
Fundamental Factors Engine — Pure compute functions for fundamental analysis.

All functions are pure: they receive data as input, produce factors as output.
No DB calls, no side effects.

Point-in-time: when computing a ratio that needs price (like P/E), the caller
must provide the closing price as of the filing_date — NOT the report_date.
This prevents look-ahead bias in backtests.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FundamentalResult:
    """Result of a single fundamental factor computation."""
    name: str
    value: Optional[float]
    raw_inputs: dict[str, float] = field(default_factory=dict)
    category: str = ""  # valuation, profitability, leverage, growth, quality


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    """Safe division that returns None for invalid inputs."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


# ============================================================================
# Valuation Factors
# ============================================================================

def pe_ratio(price: float, eps: float) -> Optional[float]:
    """Price-to-Earnings ratio."""
    return _safe_div(price, eps) if eps and eps > 0 else None

def pb_ratio(market_cap: float, book_value: float) -> Optional[float]:
    """Price-to-Book ratio."""
    return _safe_div(market_cap, book_value) if book_value and book_value > 0 else None

def ps_ratio(market_cap: float, revenue: float) -> Optional[float]:
    """Price-to-Sales ratio."""
    return _safe_div(market_cap, revenue) if revenue and revenue > 0 else None

def ev_ebitda(
    market_cap: float,
    total_debt: float,
    cash: float,
    ebitda: float,
) -> Optional[float]:
    """Enterprise Value to EBITDA."""
    ev = market_cap + total_debt - cash
    return _safe_div(ev, ebitda) if ebitda and ebitda > 0 else None

def fcf_yield(market_cap: float, free_cash_flow: float) -> Optional[float]:
    """Free Cash Flow Yield = FCF / Market Cap."""
    return _safe_div(free_cash_flow, market_cap) if market_cap and market_cap > 0 else None


# ============================================================================
# Profitability Factors
# ============================================================================

def roe(net_income: float, equity: float) -> Optional[float]:
    """Return on Equity."""
    return _safe_div(net_income, equity) if equity and equity != 0 else None

def roa(net_income: float, assets: float) -> Optional[float]:
    """Return on Assets."""
    return _safe_div(net_income, assets) if assets and assets != 0 else None

def gross_margin(revenue: float, cogs: float) -> Optional[float]:
    """Gross Profit Margin."""
    gross_profit = revenue - cogs if revenue is not None and cogs is not None else None
    return _safe_div(gross_profit, revenue) if revenue and revenue > 0 else None

def operating_margin(operating_income: float, revenue: float) -> Optional[float]:
    """Operating Margin."""
    return _safe_div(operating_income, revenue) if revenue and revenue > 0 else None

def net_margin(net_income: float, revenue: float) -> Optional[float]:
    """Net Profit Margin."""
    return _safe_div(net_income, revenue) if revenue and revenue > 0 else None

def roic(net_income: float, total_debt: float, equity: float) -> Optional[float]:
    """Return on Invested Capital."""
    invested_capital = (total_debt or 0) + (equity or 0)
    return _safe_div(net_income, invested_capital) if invested_capital and invested_capital > 0 else None


# ============================================================================
# Leverage Factors
# ============================================================================

def debt_to_equity(total_debt: float, equity: float) -> Optional[float]:
    """Debt-to-Equity ratio."""
    return _safe_div(total_debt, equity) if equity and equity != 0 else None

def debt_to_asset(total_debt: float, assets: float) -> Optional[float]:
    """Debt-to-Asset ratio."""
    return _safe_div(total_debt, assets) if assets and assets != 0 else None

def interest_coverage(operating_income: float, interest_expense: float) -> Optional[float]:
    """Interest Coverage Ratio."""
    return _safe_div(operating_income, interest_expense) if interest_expense and interest_expense != 0 else None

def current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
    """Current Ratio."""
    return _safe_div(current_assets, current_liabilities) if current_liabilities and current_liabilities != 0 else None

def quick_ratio(cash: float, ar: float, current_liabilities: float) -> Optional[float]:
    """Quick Ratio = (Cash + AR) / Current Liabilities."""
    numerator = (cash or 0) + (ar or 0)
    return _safe_div(numerator, current_liabilities) if current_liabilities and current_liabilities != 0 else None


# ============================================================================
# Growth Factors
# ============================================================================

def growth_yoy(current: float, prior: float) -> Optional[float]:
    """Year-over-Year growth rate."""
    if current is None or prior is None or prior == 0:
        return None
    return (current / prior) - 1.0

def revenue_growth_yoy(revenue: float, prior_revenue: float) -> Optional[float]:
    return growth_yoy(revenue, prior_revenue)

def earnings_growth_yoy(net_income: float, prior_net_income: float) -> Optional[float]:
    return growth_yoy(net_income, prior_net_income)

def eps_growth_yoy(eps: float, prior_eps: float) -> Optional[float]:
    return growth_yoy(eps, prior_eps)

def book_value_growth_yoy(equity: float, prior_equity: float) -> Optional[float]:
    return growth_yoy(equity, prior_equity)


# ============================================================================
# Quality Factors
# ============================================================================

def accruals_ratio(net_income: float, operating_cash_flow: float, assets: float) -> Optional[float]:
    """
    Accruals Ratio = (Net Income - Operating Cash Flow) / Total Assets

    High accruals = lower earnings quality.
    """
    if net_income is None or operating_cash_flow is None or assets is None or assets == 0:
        return None
    return float(net_income - operating_cash_flow) / float(assets)

def fcf_to_net_income(free_cash_flow: float, net_income: float) -> Optional[float]:
    """FCF / Net Income — earnings backed by cash."""
    return _safe_div(free_cash_flow, net_income) if net_income and net_income != 0 else None


# ============================================================================
# Comprehensive Fundamental Analysis
# ============================================================================

def compute_all_fundamental(
    items: dict[str, float],
    prior_items: Optional[dict[str, float]] = None,
    price: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
) -> dict[str, FundamentalResult]:
    """
    Compute all fundamental factors from XBRL items dict.

    Args:
        items: Current period XBRL items (e.g., from fundamentals.items)
        prior_items: Prior period items (for growth calculations)
        price: Closing price as of filing_date (for valuation ratios)
        shares_outstanding: Share count (for market cap)

    Returns dict of factor_name -> FundamentalResult.
    """
    results: dict[str, FundamentalResult] = {}
    prior_items = prior_items or {}

    # Extract current period values
    revenue = items.get("Revenues")
    cogs = items.get("CostOfRevenue") or items.get("CostOfGoodsAndServicesSold")
    net_income = items.get("NetIncomeLoss")
    operating_income = items.get("OperatingIncomeLoss")
    assets = items.get("Assets")
    liabilities = items.get("Liabilities")
    equity = items.get("StockholdersEquity")
    cash = items.get("CashAndCashEquivalentsAtCarryingValue")
    eps = items.get("EarningsPerShareBasic")
    shares = shares_outstanding or items.get("CommonStockSharesOutstanding")
    lt_debt = items.get("LongTermDebt")
    inventory = items.get("InventoryNet")
    ar = items.get("AccountsReceivableNetCurrent")
    current_liabilities = items.get("LiabilitiesCurrent")
    current_assets = items.get("AssetsCurrent")
    ocf = items.get("NetCashProvidedByUsedInOperatingActivities")
    capex = items.get("PaymentsToAcquirePropertyPlantAndEquipment")
    interest_exp = items.get("InterestExpense")
    ebitda = operating_income  # Simplified; would need D&A adjustment for true EBITDA

    # Market cap
    market_cap = (price * shares) if (price is not None and shares is not None) else None

    # Free cash flow
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None

    # ── Valuation ────────────────────────────────────────────────────────────
    results["pe_ratio"] = FundamentalResult("pe_ratio", pe_ratio(price, eps), {"price": price, "eps": eps}, "valuation")
    results["pb_ratio"] = FundamentalResult("pb_ratio", pb_ratio(market_cap, equity), {"market_cap": market_cap, "equity": equity}, "valuation")
    results["ps_ratio"] = FundamentalResult("ps_ratio", ps_ratio(market_cap, revenue), {"market_cap": market_cap, "revenue": revenue}, "valuation")
    results["ev_ebitda"] = FundamentalResult("ev_ebitda", ev_ebitda(market_cap, lt_debt, cash, ebitda), {"market_cap": market_cap, "debt": lt_debt, "cash": cash, "ebitda": ebitda}, "valuation")
    results["fcf_yield"] = FundamentalResult("fcf_yield", fcf_yield(market_cap, fcf), {"market_cap": market_cap, "fcf": fcf}, "valuation")

    # ── Profitability ────────────────────────────────────────────────────────
    results["roe"] = FundamentalResult("roe", roe(net_income, equity), {"net_income": net_income, "equity": equity}, "profitability")
    results["roa"] = FundamentalResult("roa", roa(net_income, assets), {"net_income": net_income, "assets": assets}, "profitability")
    results["gross_margin"] = FundamentalResult("gross_margin", gross_margin(revenue, cogs), {"revenue": revenue, "cogs": cogs}, "profitability")
    results["operating_margin"] = FundamentalResult("operating_margin", operating_margin(operating_income, revenue), {"op_income": operating_income, "revenue": revenue}, "profitability")
    results["net_margin"] = FundamentalResult("net_margin", net_margin(net_income, revenue), {"net_income": net_income, "revenue": revenue}, "profitability")
    results["roic"] = FundamentalResult("roic", roic(net_income, lt_debt, equity), {"net_income": net_income, "debt": lt_debt, "equity": equity}, "profitability")

    # ── Leverage ─────────────────────────────────────────────────────────────
    results["debt_to_equity"] = FundamentalResult("debt_to_equity", debt_to_equity(lt_debt, equity), {"debt": lt_debt, "equity": equity}, "leverage")
    results["debt_to_asset"] = FundamentalResult("debt_to_asset", debt_to_asset(liabilities, assets), {"liabilities": liabilities, "assets": assets}, "leverage")
    results["interest_coverage"] = FundamentalResult("interest_coverage", interest_coverage(operating_income, interest_exp), {"op_income": operating_income, "interest": interest_exp}, "leverage")
    results["current_ratio"] = FundamentalResult("current_ratio", current_ratio(current_assets, current_liabilities), {"current_assets": current_assets, "current_liabilities": current_liabilities}, "leverage")
    results["quick_ratio"] = FundamentalResult("quick_ratio", quick_ratio(cash, ar, current_liabilities), {"cash": cash, "ar": ar, "current_liabilities": current_liabilities}, "leverage")

    # ── Growth ───────────────────────────────────────────────────────────────
    prior_revenue = prior_items.get("Revenues")
    prior_ni = prior_items.get("NetIncomeLoss")
    prior_eps = prior_items.get("EarningsPerShareBasic")
    prior_equity = prior_items.get("StockholdersEquity")

    results["revenue_growth_yoy"] = FundamentalResult("revenue_growth_yoy", revenue_growth_yoy(revenue, prior_revenue), {"current": revenue, "prior": prior_revenue}, "growth")
    results["earnings_growth_yoy"] = FundamentalResult("earnings_growth_yoy", earnings_growth_yoy(net_income, prior_ni), {"current": net_income, "prior": prior_ni}, "growth")
    results["eps_growth_yoy"] = FundamentalResult("eps_growth_yoy", eps_growth_yoy(eps, prior_eps), {"current": eps, "prior": prior_eps}, "growth")
    results["book_value_growth_yoy"] = FundamentalResult("book_value_growth_yoy", book_value_growth_yoy(equity, prior_equity), {"current": equity, "prior": prior_equity}, "growth")

    # ── Quality ──────────────────────────────────────────────────────────────
    results["accruals_ratio"] = FundamentalResult("accruals_ratio", accruals_ratio(net_income, ocf, assets), {"ni": net_income, "ocf": ocf, "assets": assets}, "quality")
    results["fcf_to_net_income"] = FundamentalResult("fcf_to_net_income", fcf_to_net_income(fcf, net_income), {"fcf": fcf, "ni": net_income}, "quality")

    return results
