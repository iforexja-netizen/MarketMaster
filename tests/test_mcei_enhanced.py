"""
Tests for the enhanced MCEI engine.
"""

from datetime import date

import pytest

from marketmaster.engines.mcei import (
    MCEIResult,
    calculate_mcei,
    compute_component,
    _to_percentile,
    _to_zscore,
    _align_sign,
    _classify_regime,
)
from marketmaster.config.mcei_series import MCEI_COMPONENTS


class TestPercentile:
    def test_percentile_basic(self):
        history = [1, 2, 3, 4, 5]
        # Value of 3 should be roughly the 50th percentile
        p = _to_percentile(3, history)
        assert 40 <= p <= 60

    def test_percentile_max(self):
        history = [1, 2, 3, 4, 5]
        p = _to_percentile(5, history)
        assert p >= 80.0  # 4/5 values are <= max, so ~80%

    def test_percentile_min(self):
        history = [1, 2, 3, 4, 5]
        p = _to_percentile(0, history)
        assert p == 0.0

    def test_percentile_empty_history(self):
        p = _to_percentile(5, [])
        assert p == 50.0


class TestZScore:
    def test_zscore_basic(self):
        history = [1, 2, 3, 4, 5]
        z = _to_zscore(3, history)
        assert abs(z) < 0.01  # 3 is the mean

    def test_zscore_positive(self):
        history = [1, 2, 3, 4, 5]
        z = _to_zscore(5, history)
        assert z > 1.0

    def test_zscore_negative(self):
        history = [1, 2, 3, 4, 5]
        z = _to_zscore(1, history)
        assert z < -1.0

    def test_zscore_empty(self):
        z = _to_zscore(5, [])
        assert z == 0.0


class TestSignAlignment:
    def test_pos_sign_unchanged(self):
        assert _align_sign(75.0, "pos") == 75.0

    def test_neg_sign_flipped(self):
        assert _align_sign(75.0, "neg") == 25.0


class TestRegimeClassification:
    def test_strong_expansion(self):
        assert _classify_regime(85.0) == "STRONG_EXPANSION"

    def test_expansion(self):
        assert _classify_regime(65.0) == "EXPANSION"

    def test_neutral(self):
        assert _classify_regime(50.0) == "NEUTRAL"

    def test_contraction(self):
        assert _classify_regime(30.0) == "CONTRACTION"

    def test_strong_contraction(self):
        assert _classify_regime(15.0) == "STRONG_CONTRACTION"


class TestCalculateMCEI:
    def test_empty_components_returns_neutral(self):
        result = calculate_mcei({}, {})
        assert result.score == 50.0
        assert result.regime == "NEUTRAL"

    def test_single_component(self):
        # Use fed_funds_rate with high history (low value = expansionary)
        comp = MCEI_COMPONENTS[7]  # fed_funds_rate, neg sign
        values = {comp.name: 5.25}
        histories = {comp.name: [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]}
        result = calculate_mcei(values, histories, as_of_date=date(2025, 1, 15))

        assert 0 <= result.score <= 100
        assert result.regime in ["STRONG_EXPANSION", "EXPANSION", "NEUTRAL", "CONTRACTION", "STRONG_CONTRACTION"]
        assert len(result.components) == 1
        assert comp.name in result.components

    def test_all_components(self):
        """Test with all MCEI components present."""
        values = {comp.name: 50.0 for comp in MCEI_COMPONENTS}
        histories = {comp.name: [25.0, 50.0, 75.0] for comp in MCEI_COMPONENTS}
        result = calculate_mcei(values, histories)

        assert 0 <= result.score <= 100
        assert result.regime is not None
        assert len(result.components) == len(MCEI_COMPONENTS)

    def test_score_bounded(self):
        """Score should always be 0-100."""
        values = {comp.name: 999.0 for comp in MCEI_COMPONENTS}
        histories = {comp.name: [1.0] for comp in MCEI_COMPONENTS}
        result = calculate_mcei(values, histories)
        assert 0 <= result.score <= 100

    def test_to_dict(self):
        values = {MCEI_COMPONENTS[0].name: 50.0}
        histories = {MCEI_COMPONENTS[0].name: [25.0, 50.0, 75.0]}
        result = calculate_mcei(values, histories)
        d = result.to_dict()

        assert "score" in d
        assert "regime" in d
        assert "components" in d
        assert MCEI_COMPONENTS[0].name in d["components"]
