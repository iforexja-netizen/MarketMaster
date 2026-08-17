"""
Strategy Registry — Maps strategies to regimes and manages their lifecycle.

The registry is the single source of truth for available strategies.
The orchestrator queries the registry for strategies applicable to
the current market regime, then dispatches each strategy to evaluate
securities in the universe.
"""

from dataclasses import dataclass, field
from typing import Optional

from marketmaster.strategies.base import Strategy, SignalDirection, StrategyConfig
from marketmaster.strategies.strategies import create_all_strategies, get_strategies_for_regime


class StrategyRegistry:
    """
    Registry of all trading strategies.

    Maintains the complete strategy library and provides regime-based
    filtering so the orchestrator only runs applicable strategies.
    """

    def __init__(self):
        self._strategies: dict[str, Strategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all 16 built-in strategies."""
        for strategy in create_all_strategies():
            self._strategies[strategy.name] = strategy

    def register(self, strategy: Strategy):
        """Register a custom strategy."""
        if strategy.name in self._strategies:
            raise ValueError(f"Strategy already registered: {strategy.name}")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Strategy:
        """Get a strategy by name."""
        if name not in self._strategies:
            raise KeyError(f"Strategy not found: {name}")
        return self._strategies[name]

    def all(self) -> list[Strategy]:
        """Get all registered strategies."""
        return list(self._strategies.values())

    def for_regime(self, regime: str) -> list[Strategy]:
        """Get all strategies applicable to a given regime."""
        return [s for s in self._strategies.values() if s.is_applicable(regime)]

    def names(self) -> list[str]:
        """Get all strategy names."""
        return list(self._strategies.keys())

    def count(self) -> int:
        """Total number of strategies."""
        return len(self._strategies)
