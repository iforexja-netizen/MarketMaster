"""
Tests for the immutable decision log — hash chaining and tamper detection.
"""

import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from marketmaster.db.decision_log import (
    _compute_decision_hash,
    log_decision,
    verify_chain_integrity,
)


class TestDecisionHashing:
    """Test the hash computation logic."""

    def test_hash_is_deterministic(self):
        """Same inputs should produce the same hash."""
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        hash1 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy="momentum", regime="BULL",
            approved=True, score=0.85, evidence={"agent": "test"},
            risk_assessment={"max_risk": 0.5}, context={"market": "up"},
            agent_chain=[{"agent": "technical"}], prev_hash=None,
        )
        hash2 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy="momentum", regime="BULL",
            approved=True, score=0.85, evidence={"agent": "test"},
            risk_assessment={"max_risk": 0.5}, context={"market": "up"},
            agent_chain=[{"agent": "technical"}], prev_hash=None,
        )
        assert hash1 == hash2

    def test_different_inputs_produce_different_hash(self):
        """Different evidence should produce a different hash."""
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        hash1 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy=None, regime=None,
            approved=False, score=None, evidence={"bull": "case1"},
            risk_assessment={}, context={}, agent_chain=[], prev_hash=None,
        )
        hash2 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy=None, regime=None,
            approved=False, score=None, evidence={"bull": "case2"},
            risk_assessment={}, context={}, agent_chain=[], prev_hash=None,
        )
        assert hash1 != hash2

    def test_hash_is_sha256(self):
        """The hash should be a 64-character hex string (SHA-256)."""
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        h = _compute_decision_hash(
            timestamp=now, security_id=None, symbol=None,
            decision_type="signal", strategy=None, regime=None,
            approved=False, score=None, evidence={},
            risk_assessment={}, context={}, agent_chain=[], prev_hash=None,
        )
        assert len(h) == 64
        int(h, 16)  # should be valid hex

    def test_prev_hash_affects_hash(self):
        """The previous hash should affect the current hash (chaining)."""
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        h1 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy=None, regime=None,
            approved=True, score=None, evidence={},
            risk_assessment={}, context={}, agent_chain=[], prev_hash=None,
        )
        h2 = _compute_decision_hash(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy=None, regime=None,
            approved=True, score=None, evidence={},
            risk_assessment={}, context={}, agent_chain=[], prev_hash="abc123",
        )
        assert h1 != h2


class TestLogDecision:
    """Test the log_decision function with mocked DB."""

    @patch('marketmaster.db.decision_log._get_latest_hash')
    def test_first_decision_has_null_prev_hash(self, mock_get_latest):
        """The first decision in the chain should have prev_hash=None."""
        mock_get_latest.return_value = None
        db = MagicMock()
        db.query.return_value.order_by.return_value.first.return_value = None

        decision = log_decision(
            db=db,
            security_id=1,
            symbol="AAPL",
            decision_type="entry",
            approved=True,
            score=0.85,
            evidence={"agent": "test"},
            commit=False,
        )

        assert decision.prev_hash is None
        assert decision.decision_hash is not None
        assert len(decision.decision_hash) == 64
        db.add.assert_called_once()

    @patch('marketmaster.db.decision_log._get_latest_hash')
    def test_subsequent_decision_chains_to_previous(self, mock_get_latest):
        """The second decision should chain to the first."""
        first_hash = "a" * 64
        mock_get_latest.return_value = first_hash
        db = MagicMock()

        decision = log_decision(
            db=db,
            security_id=1,
            symbol="MSFT",
            decision_type="exit",
            approved=False,
            evidence={"reason": "stop_hit"},
            commit=False,
        )

        assert decision.prev_hash == first_hash
        assert decision.decision_hash != first_hash  # different content


class TestChainIntegrity:
    """Test the chain verification logic."""

    def test_verify_empty_chain(self):
        """An empty chain should be valid."""
        db = MagicMock()
        db.query.return_value.order_by.return_value.all.return_value = []
        is_valid, errors = verify_chain_integrity(db)
        assert is_valid
        assert errors == []
