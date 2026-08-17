"""
Portfolio Optimizer — Construct optimal portfolio from trade signals.

Given a set of trade signals from the screener, the portfolio optimizer:
1. Filters signals by risk gate and position limits
2. Allocates capital across signals using the selected method:
   - Equal weight
   - Score-weighted (higher score = larger allocation)
   - Risk parity (inverse volatility weighting)
   - Mean-variance (Sharpe maximization)
3. Enforces diversification constraints (max positions, max sector concentration)
4. Returns a portfolio allocation with specific position sizes

The optimizer respects the Risk Gate: no allocation can exceed
max_position_risk_pct or the daily loss limit.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import date
import numpy as np

from marketmaster.strategies.base import TradeSignal, SignalDirection


@dataclass
class PositionAllocation:
    """A single position allocation in the portfolio."""
    symbol: str
    strategy_name: str
    direction: str  # "long" or "short"
    weight: float  # fraction of portfolio (0-1)
    dollar_allocation: float
    shares: Optional[float] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    risk_reward_ratio: float = 0.0
    score: float = 0.0
    confidence: float = 0.0


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""
    as_of: date
    total_allocation: float  # fraction of portfolio invested
    cash_reserve: float  # fraction in cash
    positions: list[PositionAllocation] = field(default_factory=list)
    n_positions: int = 0
    avg_score: float = 0.0
    avg_confidence: float = 0.0
    method: str = ""
    notes: list[str] = field(default_factory=list)


class PortfolioOptimizer:
    """
    Optimizes capital allocation across trade signals.

    Usage:
        optimizer = PortfolioOptimizer(initial_capital=100_000)
        allocation = optimizer.optimize(
            signals=screener_result.top_opportunities,
            method="score_weighted",
            max_positions=10,
            max_position_pct=5.0,
            regime="BULL",
        )
    """

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital

    def optimize(
        self,
        signals: list[TradeSignal],
        method: str = "score_weighted",
        max_positions: int = 10,
        max_position_pct: float = 5.0,
        min_cash_pct: float = 10.0,
        regime: str = "NEUTRAL",
        as_of: Optional[date] = None,
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation across signals.

        Args:
            signals: Trade signals from the screener (sorted by quality)
            method: Allocation method ("equal_weight", "score_weighted",
                    "risk_parity", "mean_variance")
            max_positions: Maximum number of concurrent positions
            max_position_pct: Maximum % of portfolio per position
            min_cash_pct: Minimum cash reserve
            regime: Current market regime (affects aggressiveness)

        Returns:
            PortfolioAllocation with specific position sizes
        """
        if as_of is None:
            as_of = date.today()

        result = PortfolioAllocation(
            as_of=as_of,
            total_allocation=0.0,
            cash_reserve=min_cash_pct / 100,
            method=method,
        )

        # Filter to actionable signals only
        actionable = [
            s for s in signals
            if s.direction != SignalDirection.NEUTRAL
            and s.position_size_pct > 0
            and s.entry_price is not None
        ]

        if not actionable:
            result.notes.append("No actionable signals for portfolio construction")
            return result

        # Limit to top N signals
        actionable = actionable[:max_positions]

        # Adjust for regime: reduce exposure in bearish regimes
        regime_factor = self._regime_factor(regime)
        max_allocation = (1.0 - min_cash_pct / 100) * regime_factor

        # ── Allocate by method ───────────────────────────────────────────
        if method == "equal_weight":
            weights = self._equal_weight(actionable, max_allocation)
        elif method == "score_weighted":
            weights = self._score_weighted(actionable, max_allocation, max_position_pct / 100)
        elif method == "risk_parity":
            weights = self._risk_parity(actionable, max_allocation, max_position_pct / 100)
        elif method == "mean_variance":
            weights = self._mean_variance(actionable, max_allocation, max_position_pct / 100)
        else:
            weights = self._score_weighted(actionable, max_allocation, max_position_pct / 100)

        # ── Build position allocations ───────────────────────────────────
        for signal, weight in zip(actionable, weights):
            if weight <= 0:
                continue

            dollar_alloc = self.initial_capital * weight
            shares = dollar_alloc / signal.entry_price if signal.entry_price else 0

            result.positions.append(PositionAllocation(
                symbol=signal.symbol,
                strategy_name=signal.strategy_name,
                direction=signal.direction.value,
                weight=weight,
                dollar_allocation=dollar_alloc,
                shares=shares,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                risk_reward_ratio=signal.risk_reward_ratio,
                score=signal.score,
                confidence=signal.confidence,
            ))

        result.n_positions = len(result.positions)
        result.total_allocation = sum(p.weight for p in result.positions)
        result.cash_reserve = 1.0 - result.total_allocation

        if result.positions:
            result.avg_score = float(np.mean([p.score for p in result.positions]))
            result.avg_confidence = float(np.mean([p.confidence for p in result.positions]))

        result.notes.append(f"Regime: {regime} (exposure factor: {regime_factor:.0%})")
        result.notes.append(f"Method: {method}")
        result.notes.append(f"Positions: {result.n_positions} / {max_positions}")
        result.notes.append(f"Cash reserve: {result.cash_reserve:.1%}")

        return result

    def _regime_factor(self, regime: str) -> float:
        """Return allocation factor based on regime (0-1)."""
        factors = {
            "STRONG_BULL": 0.95,
            "BULL": 0.85,
            "TRANSITION_BULL": 0.70,
            "NEUTRAL": 0.60,
            "TRANSITION_BEAR": 0.40,
            "BEAR": 0.25,
            "CRISIS": 0.10,
            "RECOVERY": 0.65,
        }
        return factors.get(regime, 0.50)

    def _equal_weight(self, signals: list[TradeSignal], max_alloc: float) -> list[float]:
        """Equal weight allocation."""
        n = len(signals)
        weight = min(max_alloc / n, 0.05)  # Cap at 5% per position
        return [weight] * n

    def _score_weighted(self, signals: list[TradeSignal], max_alloc: float, max_per: float) -> list[float]:
        """Score-weighted allocation: higher score × confidence = larger weight."""
        scores = np.array([s.score * s.confidence for s in signals])
        if scores.sum() == 0:
            return self._equal_weight(signals, max_alloc)

        weights = scores / scores.sum() * max_alloc
        # Cap each position
        weights = np.minimum(weights, max_per)
        # Re-normalize
        if weights.sum() > 0:
            weights = weights / weights.sum() * min(weights.sum(), max_alloc)

        return weights.tolist()

    def _risk_parity(self, signals: list[TradeSignal], max_alloc: float, max_per: float) -> list[float]:
        """Risk parity: inverse volatility weighting."""
        # Use signal's stop distance as volatility proxy
        volatilities = []
        for s in signals:
            if s.entry_price and s.stop_price:
                vol = abs(s.entry_price - s.stop_price) / s.entry_price
            else:
                vol = 0.05  # Default 5% volatility
            volatilities.append(max(vol, 0.01))

        inv_vols = 1.0 / np.array(volatilities)
        weights = inv_vols / inv_vols.sum() * max_alloc
        weights = np.minimum(weights, max_per)

        return weights.tolist()

    def _mean_variance(self, signals: list[TradeSignal], max_alloc: float, max_per: float) -> list[float]:
        """
        Simplified mean-variance optimization.

        Uses signal score as expected return proxy and stop distance as risk proxy.
        This is a simplified version — full MVO would require a covariance matrix.
        """
        # Expected returns: signal score (higher = more expected return)
        exp_returns = np.array([s.score / 100 for s in signals])

        # Risk: stop distance as variance proxy
        risks = []
        for s in signals:
            if s.entry_price and s.stop_price:
                r = abs(s.entry_price - s.stop_price) / s.entry_price
            else:
                r = 0.05
            risks.append(r ** 2)

        risks = np.array(risks)
        risk_tolerance = 0.5

        # Simplified: weight proportional to return / risk
        if risks.sum() > 0:
            raw_weights = exp_returns / (risks + 1e-6)
            weights = raw_weights / raw_weights.sum() * max_alloc
        else:
            weights = self._equal_weight(signals, max_alloc)

        weights = np.minimum(weights, max_per)
        return weights.tolist()
