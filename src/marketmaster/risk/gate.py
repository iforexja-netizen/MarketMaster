from dataclasses import dataclass

@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str]

def risk_gate(position_risk_pct, max_position_risk_pct, daily_loss_pct,
               max_daily_loss_pct, live_trading_enabled):
    reasons = []
    if position_risk_pct > max_position_risk_pct: reasons.append("POSITION_RISK_LIMIT")
    if daily_loss_pct >= max_daily_loss_pct: reasons.append("DAILY_LOSS_LIMIT")
    if not live_trading_enabled: reasons.append("LIVE_TRADING_DISABLED")
    return RiskDecision(len(reasons) == 0, reasons)
