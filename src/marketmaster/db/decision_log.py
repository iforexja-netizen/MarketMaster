"""
Decision Log — Immutable, Hash-Chained Decision Recording

The decision log is the system's memory. Every decision is:
1. Append-only — cannot be updated or deleted (enforced by DB triggers)
2. Hash-chained — each decision references the previous decision's hash
3. Tamper-evident — changing any historical decision breaks the chain

This module provides the logic for creating decision records with proper hashing.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from marketmaster.db.models import Decision


def _compute_decision_hash(
    timestamp: datetime,
    security_id: Optional[int],
    symbol: Optional[str],
    decision_type: str,
    strategy: Optional[str],
    regime: Optional[str],
    approved: bool,
    score: Optional[float],
    evidence: dict[str, Any],
    risk_assessment: dict[str, Any],
    context: dict[str, Any],
    agent_chain: list[dict[str, Any]],
    prev_hash: Optional[str],
) -> str:
    """
    Compute SHA-256 hash of a decision's content.

    The hash includes all substantive fields plus the previous decision's hash.
    This creates a tamper-evident chain: modifying any historical decision
    changes its hash, which breaks the chain for all subsequent decisions.
    """
    content = {
        "timestamp": timestamp.isoformat(),
        "security_id": security_id,
        "symbol": symbol,
        "decision_type": decision_type,
        "strategy": strategy,
        "regime": regime,
        "approved": approved,
        "score": score,
        "evidence": evidence,
        "risk_assessment": risk_assessment,
        "context": context,
        "agent_chain": agent_chain,
        "prev_hash": prev_hash,
    }
    # Sort keys for deterministic hashing
    serialized = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_latest_hash(db: Session) -> Optional[str]:
    """Get the hash of the most recent decision in the chain."""
    latest = db.query(Decision).order_by(Decision.id.desc()).first()
    return latest.decision_hash if latest else None


def log_decision(
    db: Session,
    security_id: Optional[int],
    symbol: Optional[str],
    decision_type: str,
    strategy: Optional[str] = None,
    regime: Optional[str] = None,
    approved: bool = False,
    score: Optional[float] = None,
    expected_value: Optional[float] = None,
    evidence: Optional[dict[str, Any]] = None,
    risk_assessment: Optional[dict[str, Any]] = None,
    context: Optional[dict[str, Any]] = None,
    agent_chain: Optional[list[dict[str, Any]]] = None,
    human_approved: bool = False,
    human_approver: Optional[str] = None,
    commit: bool = True,
) -> Decision:
    """
    Append a new decision to the immutable log.

    This function:
    1. Fetches the latest decision's hash (chain link)
    2. Computes the new decision's hash
    3. Inserts the decision (append-only — DB triggers prevent modification)

    Args:
        db: Database session
        security_id: FK to security_master.id (nullable for market-wide decisions)
        symbol: Ticker symbol (for quick reference without joins)
        decision_type: signal, entry, exit, adjustment, reject, review
        strategy: Strategy name if applicable
        regime: Current market regime at decision time
        approved: Whether the risk gate approved this decision
        score: Opportunity score (0-1)
        expected_value: Expected value of the trade
        evidence: Full agent evidence (JSONB)
        risk_assessment: Risk gate output (JSONB)
        context: Market state, MCEI, regime at decision time (JSONB)
        agent_chain: Ordered list of agents that participated
        human_approved: Whether a human approved this
        human_approver: Who approved it
        commit: Whether to commit the transaction

    Returns:
        The created Decision record
    """
    now = datetime.now(timezone.utc)

    evidence = evidence or {}
    risk_assessment = risk_assessment or {}
    context = context or {}
    agent_chain = agent_chain or []

    prev_hash = _get_latest_hash(db)

    decision_hash = _compute_decision_hash(
        timestamp=now,
        security_id=security_id,
        symbol=symbol,
        decision_type=decision_type,
        strategy=strategy,
        regime=regime,
        approved=approved,
        score=score,
        evidence=evidence,
        risk_assessment=risk_assessment,
        context=context,
        agent_chain=agent_chain,
        prev_hash=prev_hash,
    )

    decision = Decision(
        decision_hash=decision_hash,
        prev_hash=prev_hash,
        timestamp=now,
        security_id=security_id,
        symbol=symbol,
        decision_type=decision_type,
        strategy=strategy,
        regime=regime,
        approved=approved,
        score=score,
        expected_value=expected_value,
        evidence=evidence,
        risk_assessment=risk_assessment,
        context=context,
        agent_chain=agent_chain,
        human_approved=human_approved,
        human_approver=human_approver,
        approved_at=now if human_approved else None,
    )

    db.add(decision)
    if commit:
        db.commit()
        db.refresh(decision)

    return decision


def verify_chain_integrity(db: Session) -> tuple[bool, list[str]]:
    """
    Verify the integrity of the decision hash chain.

    Recomputes each decision's hash and checks that:
    1. Each decision's hash matches its content
    2. Each decision's prev_hash matches the previous decision's hash

    Returns:
        (is_valid, errors) — True if chain is intact, plus list of any errors
    """
    decisions = db.query(Decision).order_by(Decision.id.asc()).all()
    errors: list[str] = []
    expected_prev_hash: Optional[str] = None

    for d in decisions:
        if d.prev_hash != expected_prev_hash:
            errors.append(
                f"Decision {d.id}: prev_hash mismatch. "
                f"Expected {expected_prev_hash}, got {d.prev_hash}"
            )

        recomputed = _compute_decision_hash(
            timestamp=d.timestamp,
            security_id=d.security_id,
            symbol=d.symbol,
            decision_type=d.decision_type,
            strategy=d.strategy,
            regime=d.regime,
            approved=d.approved,
            score=d.score,
            evidence=d.evidence,
            risk_assessment=d.risk_assessment,
            context=d.context,
            agent_chain=d.agent_chain,
            prev_hash=d.prev_hash,
        )

        if recomputed != d.decision_hash:
            errors.append(
                f"Decision {d.id}: hash mismatch. "
                f"Expected {recomputed}, stored {d.decision_hash}"
            )

        expected_prev_hash = d.decision_hash

    return len(errors) == 0, errors
