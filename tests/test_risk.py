from marketmaster.risk.gate import risk_gate

def test_risk_gate_blocks_live_trading_by_default():
    result = risk_gate(.002, .005, .001, .02, False)
    assert not result.approved
    assert "LIVE_TRADING_DISABLED" in result.reasons
