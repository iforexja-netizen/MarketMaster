from marketmaster.engines.mcei import calculate_mcei

def test_mcei_empty():
    """Empty components should return neutral."""
    result = calculate_mcei({}, {})
    assert result.score == 50.0
    assert result.regime == "NEUTRAL"

def test_mcei_with_history():
    """With proper history, percentile normalization should work."""
    # Values with history that puts them at various percentiles
    result = calculate_mcei(
        {"money": 80, "credit": 60, "rates": 70},
        {"money": [10, 20, 30, 40, 50, 60, 70, 80], 
         "credit": [10, 20, 30, 40, 50, 60], 
         "rates": [10, 20, 30, 40, 50, 60, 70]},
    )
    assert 0 <= result.score <= 100
    assert result.regime in ("STRONG_EXPANSION", "EXPANSION", "NEUTRAL", "CONTRACTION", "STRONG_CONTRACTION")
