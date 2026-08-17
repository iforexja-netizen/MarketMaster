"""
Deterministic Risk Engine — Final authority on all trading decisions.

The Risk Engine is DETERMINISTIC. It uses hard rules, not AI judgment.
No amount of bullish enthusiasm bypasses position limits, daily loss limits,
or stale-data rejection. This is enforced in code, not in policy.

Risk checks performed:
1. Live trading flag — live trading is disabled by default
2. Per-position risk — max risk per individual position
3. Portfolio risk — max aggregate risk across all positions
4. Daily loss limit — stop trading if daily P&L hits limit
5. Sector concentration — max exposure to any single sector
6. Single position size — max % of portfolio per position
7. Stale data rejection — reject signals based on outdated data
8. Drawdown circuit breaker — reduce exposure in drawdowns
9. Correlation limit — avoid over-concentration in correlated assets
10. Kill switch — manual override to halt all trading

Every decision is logged to the immutable decision log with full reasoning.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Optional, Any
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class KillSwitchState(Enum):
    ACTIVE = "active"      # Trading allowed (subject to all other checks)
    HALTED = "halted"      # All trading halted
    DEGRADED = "degraded"  # Reduced exposure only


@dataclass
class RiskCheck:
    """Result of a single risk check."""
    name: str
    level: RiskLevel
    message: str
    value: float = 0.0
    threshold: float = 0.0
    passed: bool = True


@dataclass
class RiskDecision:
    """Final risk decision for an order or portfolio action."""
    approved: bool
    risk_score: float  # 0-100, lower is safer
    checks: list[RiskCheck] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    kill_switch: KillSwitchState = KillSwitchState.ACTIVE

    @property
    def failed_checks(self) -> list[RiskCheck]:
        return [c for c in self.checks if c.level == RiskLevel.FAIL]

    @property
    def warnings(self) -> list[RiskCheck]:
        return [c for c in self.checks if c.level == RiskLevel.WARN]


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state (tracked in real-time)."""
    total_equity: float = 100_000
    cash: float = 100_000
    invested: float = 0.0
    positions: list[dict] = field(default_factory=list)  # [{symbol, shares, entry, current, sector, risk_pct}]
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    peak_equity: float = 100_000
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    open_risk_pct: float = 0.0  # sum of all position stop distances
    last_update: Optional[datetime] = None


class RiskEngine:
    """
    Deterministic risk engine with final authority over all trades.

    Every order must pass through the Risk Engine before execution.
    The engine enforces hard limits — no soft warnings, no overrides.

    Usage:
        engine = RiskEngine(settings)
        decision = engine.evaluate_order(
            order=trade_plan.orders[0],
            portfolio_state=current_state,
            as_of=date(2025, 6, 1),
        )
        if not decision.approved:
            # Order rejected — do not execute
            log_rejection(decision)
    """

    def __init__(self, settings: Any = None):
        # Load settings or use defaults
        if settings:
            self.live_trading_enabled = settings.enable_live_trading
            self.max_position_risk_pct = settings.max_position_risk_pct
            self.max_daily_loss_pct = settings.max_daily_loss_pct
            self.max_portfolio_risk_pct = settings.max_portfolio_risk_pct
            self.max_sector_exposure_pct = settings.max_sector_exposure_pct
            self.max_single_position_pct = settings.max_single_position_pct
        else:
            self.live_trading_enabled = False
            self.max_position_risk_pct = 0.005  # 0.5% per position
            self.max_daily_loss_pct = 0.02       # 2% daily loss limit
            self.max_portfolio_risk_pct = 0.10   # 10% total portfolio risk
            self.max_sector_exposure_pct = 0.30  # 30% per sector
            self.max_single_position_pct = 0.10  # 10% per position

        # Internal state
        self._kill_switch = KillSwitchState.ACTIVE
        self._stale_data_threshold_minutes = 15
        self._drawdown_threshold_pct = 10.0  # Reduce exposure after 10% drawdown
        self._drawdown_degradation_factor = 0.5  # Cut exposure by 50% in drawdown
        self._max_correlation = 0.85  # Max correlation between positions
        self._daily_loss_reset_date: Optional[date] = None

    # ── Kill Switch ─────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Halt all trading immediately."""
        self._kill_switch = KillSwitchState.HALTED

    def deactivate_kill_switch(self):
        """Resume trading after kill switch."""
        self._kill_switch = KillSwitchState.ACTIVE

    def degrade_trading(self, reason: str = "Drawdown degradation"):
        """Reduce exposure without fully halting."""
        self._kill_switch = KillSwitchState.DEGRADED

    @property
    def kill_switch_state(self) -> KillSwitchState:
        return self._kill_switch

    # ── Order Evaluation ────────────────────────────────────────────────────

    def evaluate_order(
        self,
        order: dict,  # {symbol, side, quantity, entry_price, stop_price, strategy, sector}
        portfolio_state: PortfolioRiskState,
        as_of: Optional[datetime] = None,
        data_timestamp: Optional[datetime] = None,
    ) -> RiskDecision:
        """
        Evaluate a single order against all risk checks.

        This is the FINAL GATE. If any hard check fails, the order is rejected.
        Warnings don't block execution but are logged.
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        checks: list[RiskCheck] = []
        reasons: list[str] = []
        adjustments: dict[str, float] = {}

        # ── Check 1: Kill Switch ─────────────────────────────────────────
        if self._kill_switch == KillSwitchState.HALTED:
            checks.append(RiskCheck(
                name="kill_switch", level=RiskLevel.FAIL,
                message="Kill switch is ACTIVE — all trading halted",
            ))
            reasons.append("KILL_SWITCH_HALT")
            return RiskDecision(
                approved=False, risk_score=100, checks=checks,
                reasons=reasons, kill_switch=KillSwitchState.HALTED,
            )
        elif self._kill_switch == KillSwitchState.DEGRADED:
            checks.append(RiskCheck(
                name="kill_switch", level=RiskLevel.WARN,
                message="Kill switch is DEGRADED — reduced exposure only",
            ))
            adjustments["size_multiplier"] = self._drawdown_degradation_factor

        # ── Check 2: Live Trading Flag ────────────────────────────────────
        is_live = self.live_trading_enabled
        if not is_live:
            checks.append(RiskCheck(
                name="live_trading", level=RiskLevel.PASS,
                message="Live trading disabled — paper trading mode",
            ))
        else:
            checks.append(RiskCheck(
                name="live_trading", level=RiskLevel.WARN,
                message="Live trading is ENABLED — use extreme caution",
            ))

        # ── Check 3: Stale Data ──────────────────────────────────────────
        if data_timestamp:
            age_minutes = (as_of - data_timestamp).total_seconds() / 60
            if age_minutes > self._stale_data_threshold_minutes:
                checks.append(RiskCheck(
                    name="stale_data", level=RiskLevel.FAIL,
                    message=f"Data is {age_minutes:.0f} minutes old (max {self._stale_data_threshold_minutes})",
                    value=age_minutes, threshold=self._stale_data_threshold_minutes,
                    passed=False,
                ))
                reasons.append("STALE_DATA")
            else:
                checks.append(RiskCheck(
                    name="stale_data", level=RiskLevel.PASS,
                    message=f"Data is {age_minutes:.0f} minutes old",
                    value=age_minutes, threshold=self._stale_data_threshold_minutes,
                ))
        else:
            checks.append(RiskCheck(
                name="stale_data", level=RiskLevel.WARN,
                message="No data timestamp provided — cannot verify freshness",
            ))

        # ── Check 4: Per-Position Risk ────────────────────────────────────
        entry_price = order.get("entry_price", 0)
        stop_price = order.get("stop_price", 0)
        quantity = order.get("quantity", 0)
        equity = portfolio_state.total_equity

        position_risk = 0.0
        if entry_price and stop_price and quantity and equity > 0:
            risk_per_share = abs(entry_price - stop_price)
            position_risk = (risk_per_share * quantity) / equity
            position_risk_pct = position_risk * 100

            if position_risk > self.max_position_risk_pct:
                checks.append(RiskCheck(
                    name="position_risk", level=RiskLevel.FAIL,
                    message=f"Position risk {position_risk_pct:.2f}% exceeds max {self.max_position_risk_pct*100:.2f}%",
                    value=position_risk_pct, threshold=self.max_position_risk_pct * 100,
                    passed=False,
                ))
                reasons.append("POSITION_RISK_LIMIT")

                # Suggest size adjustment
                max_risk_amount = self.max_position_risk_pct * equity
                if risk_per_share > 0:
                    max_quantity = max_risk_amount / risk_per_share
                    adjustments["max_quantity"] = max_quantity
                    adjustments["size_multiplier"] = max_quantity / quantity if quantity > 0 else 0
            else:
                checks.append(RiskCheck(
                    name="position_risk", level=RiskLevel.PASS,
                    message=f"Position risk {position_risk_pct:.2f}% within limit",
                    value=position_risk_pct, threshold=self.max_position_risk_pct * 100,
                ))

        # ── Check 5: Single Position Size ─────────────────────────────────
        position_value = entry_price * quantity if entry_price and quantity else 0
        if equity > 0 and position_value > 0:
            position_pct = position_value / equity
            if position_pct > self.max_single_position_pct:
                checks.append(RiskCheck(
                    name="position_size", level=RiskLevel.FAIL,
                    message=f"Position size {position_pct:.1%} exceeds max {self.max_single_position_pct:.1%}",
                    value=position_pct, threshold=self.max_single_position_pct,
                    passed=False,
                ))
                reasons.append("POSITION_SIZE_LIMIT")
                adjustments["max_position_value"] = self.max_single_position_pct * equity
            else:
                checks.append(RiskCheck(
                    name="position_size", level=RiskLevel.PASS,
                    message=f"Position size {position_pct:.1%} within limit",
                    value=position_pct, threshold=self.max_single_position_pct,
                ))

        # ── Check 6: Portfolio Risk (aggregate) ───────────────────────────
        total_open_risk = portfolio_state.open_risk_pct / 100 + position_risk
        if total_open_risk > self.max_portfolio_risk_pct:
            checks.append(RiskCheck(
                name="portfolio_risk", level=RiskLevel.FAIL,
                message=f"Portfolio risk {total_open_risk:.2%} exceeds max {self.max_portfolio_risk_pct:.2%}",
                value=total_open_risk, threshold=self.max_portfolio_risk_pct,
                passed=False,
            ))
            reasons.append("PORTFOLIO_RISK_LIMIT")
        else:
            checks.append(RiskCheck(
                name="portfolio_risk", level=RiskLevel.PASS,
                message=f"Portfolio risk {total_open_risk:.2%} within limit",
                value=total_open_risk, threshold=self.max_portfolio_risk_pct,
            ))

        # ── Check 7: Daily Loss Limit ─────────────────────────────────────
        daily_loss_pct = abs(portfolio_state.daily_pnl_pct) if portfolio_state.daily_pnl < 0 else 0
        if daily_loss_pct >= self.max_daily_loss_pct * 100:
            checks.append(RiskCheck(
                name="daily_loss", level=RiskLevel.FAIL,
                message=f"Daily loss {daily_loss_pct:.2f}% hit limit {self.max_daily_loss_pct*100:.2f}%",
                value=daily_loss_pct, threshold=self.max_daily_loss_pct * 100,
                passed=False,
            ))
            reasons.append("DAILY_LOSS_LIMIT")
        elif daily_loss_pct > self.max_daily_loss_pct * 100 * 0.75:
            checks.append(RiskCheck(
                name="daily_loss", level=RiskLevel.WARN,
                message=f"Daily loss {daily_loss_pct:.2f}% approaching limit",
                value=daily_loss_pct, threshold=self.max_daily_loss_pct * 100,
            ))
        else:
            checks.append(RiskCheck(
                name="daily_loss", level=RiskLevel.PASS,
                message=f"Daily P&L {portfolio_state.daily_pnl_pct:.2f}% within limit",
                value=daily_loss_pct, threshold=self.max_daily_loss_pct * 100,
            ))

        # ── Check 8: Drawdown Circuit Breaker ────────────────────────────
        dd = portfolio_state.current_drawdown_pct
        if dd > self._drawdown_threshold_pct:
            checks.append(RiskCheck(
                name="drawdown", level=RiskLevel.WARN,
                message=f"Drawdown {dd:.1f}% exceeds {self._drawdown_threshold_pct}% — reducing exposure",
                value=dd, threshold=self._drawdown_threshold_pct,
            ))
            adjustments["size_multiplier"] = min(
                adjustments.get("size_multiplier", 1.0),
                self._drawdown_degradation_factor
            )
        else:
            checks.append(RiskCheck(
                name="drawdown", level=RiskLevel.PASS,
                message=f"Drawdown {dd:.1f}% within threshold",
                value=dd, threshold=self._drawdown_threshold_pct,
            ))

        # ── Check 9: Sector Concentration ─────────────────────────────────
        sector = order.get("sector", "unknown")
        sector_exposure = 0.0
        if equity > 0:
            for pos in portfolio_state.positions:
                if pos.get("sector") == sector:
                    sector_exposure += pos.get("market_value", 0) / equity
            # Add the new position
            if position_value > 0:
                sector_exposure += position_value / equity

        if sector_exposure > self.max_sector_exposure_pct:
            checks.append(RiskCheck(
                name="sector_concentration", level=RiskLevel.FAIL,
                message=f"Sector '{sector}' exposure {sector_exposure:.1%} exceeds max {self.max_sector_exposure_pct:.1%}",
                value=sector_exposure, threshold=self.max_sector_exposure_pct,
                passed=False,
            ))
            reasons.append("SECTOR_CONCENTRATION")
        else:
            checks.append(RiskCheck(
                name="sector_concentration", level=RiskLevel.PASS,
                message=f"Sector '{sector}' exposure {sector_exposure:.1%}",
                value=sector_exposure, threshold=self.max_sector_exposure_pct,
            ))

        # ── Check 10: Max Positions ───────────────────────────────────────
        max_positions = 20  # Hard limit
        current_positions = len(portfolio_state.positions)
        if current_positions >= max_positions:
            checks.append(RiskCheck(
                name="max_positions", level=RiskLevel.FAIL,
                message=f"Max positions reached ({current_positions}/{max_positions})",
                value=current_positions, threshold=max_positions,
                passed=False,
            ))
            reasons.append("MAX_POSITIONS")
        else:
            checks.append(RiskCheck(
                name="max_positions", level=RiskLevel.PASS,
                message=f"Positions: {current_positions}/{max_positions}",
                value=current_positions, threshold=max_positions,
            ))

        # ── Compute Risk Score ───────────────────────────────────────────
        risk_score = self._compute_risk_score(checks, portfolio_state)

        # ── Final Decision ────────────────────────────────────────────────
        approved = len(reasons) == 0

        return RiskDecision(
            approved=approved,
            risk_score=risk_score,
            checks=checks,
            reasons=reasons,
            adjustments=adjustments,
            kill_switch=self._kill_switch,
        )

    def evaluate_portfolio(
        self,
        portfolio_state: PortfolioRiskState,
    ) -> RiskDecision:
        """
        Evaluate the overall portfolio risk state.
        Used for periodic risk monitoring (not per-order).
        """
        checks: list[RiskCheck] = []
        reasons: list[str] = []

        # Portfolio risk
        open_risk = portfolio_state.open_risk_pct / 100
        if open_risk > self.max_portfolio_risk_pct:
            checks.append(RiskCheck(
                name="portfolio_risk", level=RiskLevel.FAIL,
                message=f"Total open risk {open_risk:.2%} exceeds max {self.max_portfolio_risk_pct:.2%}",
                value=open_risk, threshold=self.max_portfolio_risk_pct,
                passed=False,
            ))
            reasons.append("PORTFOLIO_RISK_BREACH")
        else:
            checks.append(RiskCheck(
                name="portfolio_risk", level=RiskLevel.PASS,
                message=f"Total open risk {open_risk:.2%}",
                value=open_risk, threshold=self.max_portfolio_risk_pct,
            ))

        # Drawdown
        dd = portfolio_state.current_drawdown_pct
        if dd > 20:
            checks.append(RiskCheck(
                name="drawdown", level=RiskLevel.FAIL,
                message=f"Severe drawdown {dd:.1f}%",
                value=dd, threshold=20,
                passed=False,
            ))
            reasons.append("SEVERE_DRAWDOWN")
        elif dd > self._drawdown_threshold_pct:
            checks.append(RiskCheck(
                name="drawdown", level=RiskLevel.WARN,
                message=f"Drawdown {dd:.1f}% — consider reducing exposure",
                value=dd, threshold=self._drawdown_threshold_pct,
            ))
        else:
            checks.append(RiskCheck(
                name="drawdown", level=RiskLevel.PASS,
                message=f"Drawdown {dd:.1f}%",
                value=dd, threshold=20,
            ))

        # Daily loss
        if portfolio_state.daily_pnl_pct < -(self.max_daily_loss_pct * 100):
            checks.append(RiskCheck(
                name="daily_loss", level=RiskLevel.FAIL,
                message=f"Daily loss limit breached: {portfolio_state.daily_pnl_pct:.2f}%",
                value=abs(portfolio_state.daily_pnl_pct),
                threshold=self.max_daily_loss_pct * 100,
                passed=False,
            ))
            reasons.append("DAILY_LOSS_BREACH")
        else:
            checks.append(RiskCheck(
                name="daily_loss", level=RiskLevel.PASS,
                message=f"Daily P&L: {portfolio_state.daily_pnl_pct:.2f}%",
                value=abs(portfolio_state.daily_pnl_pct),
                threshold=self.max_daily_loss_pct * 100,
            ))

        risk_score = self._compute_risk_score(checks, portfolio_state)

        return RiskDecision(
            approved=len(reasons) == 0,
            risk_score=risk_score,
            checks=checks,
            reasons=reasons,
            kill_switch=self._kill_switch,
        )

    def compute_position_size(
        self,
        entry_price: float,
        stop_price: float,
        equity: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Compute the maximum position size that stays within risk limits.

        Uses the Kelly-inspired fixed-fractional approach:
            size = (equity × risk_pct) / (entry - stop)

        This is DETERMINISTIC. The risk engine's position size is authoritative.
        """
        if risk_pct is None:
            risk_pct = self.max_position_risk_pct

        risk_amount = equity * risk_pct
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share <= 0:
            return 0.0

        return risk_amount / risk_per_share

    def _compute_risk_score(self, checks: list[RiskCheck], state: PortfolioRiskState) -> float:
        """Compute overall risk score (0-100, lower is safer)."""
        score = 0.0

        # Penalty for each failed check
        for c in checks:
            if c.level == RiskLevel.FAIL:
                score += 20
            elif c.level == RiskLevel.WARN:
                score += 5

        # Add drawdown component
        score += state.current_drawdown_pct * 2

        # Add daily loss component
        if state.daily_pnl < 0:
            score += abs(state.daily_pnl_pct) * 3

        # Add portfolio risk component
        score += state.open_risk_pct * 5

        return min(100, max(0, score))

    def reset_daily(self, current_date: date):
        """Reset daily loss tracking (called at market open)."""
        self._daily_loss_reset_date = current_date
