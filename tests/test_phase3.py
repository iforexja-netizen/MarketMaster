"""
Phase 3 tests — Specialist agents, bull/bear debate, NLP, knowledge retrieval.

Tests the computation and analysis logic without requiring a database connection.
Uses mock DataPlane objects for agent tests.
"""

import pytest
from datetime import date, datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


# ============================================================================
# Mock DataPlane for agent tests
# ============================================================================

@dataclass
class MockMceiResult:
    score: float = 65.0
    regime: str = "BULL"
    as_of_date: date = date(2025, 6, 1)
    components: dict = field(default_factory=lambda: {
        "money_supply": 70, "credit_impulse": 60, "liquidity": 65,
        "real_rates": 55, "yield_curve": 50, "financial_conditions": 68,
        "dollar_strength": 45, "commodity_pressure": 35,
    })


@dataclass
class MockRegimeResult:
    regime: str = "BULL"
    confidence: float = 0.75
    as_of_date: date = date(2025, 6, 1)


@dataclass
class MockBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class MockSecurity:
    id: int = 1
    symbol: str = "AAPL"
    name: str = "Apple Inc"
    asset_class: str = "equity"


@dataclass
class MockFundamental:
    security_id: int = 1
    report_date: date = date(2025, 3, 31)
    period_type: str = "quarterly"
    filing_date: date = date(2025, 4, 28)
    items: dict = field(default_factory=dict)
    source: str = "sec_edgar"


def make_mock_plane(
    bars=None,
    fundamentals=None,
    mcei=MockMceiResult(),
    regime=MockRegimeResult(),
    security=None,
):
    """Create a mock DataPlane for agent testing."""
    plane = MagicMock()
    plane.get_security_by_symbol.return_value = security or MockSecurity()
    plane.get_security_by_id.return_value = security or MockSecurity()
    plane.get_latest_price.return_value = MockBar(
        date=date(2025, 6, 1), open=190, high=192, low=188, close=191, volume=50000000
    )
    plane.get_ohlcv_daily.return_value = bars or []
    plane.get_fundamentals.return_value = fundamentals or []
    plane.get_latest_mcei.return_value = mcei
    plane.get_latest_regime.return_value = regime
    return plane


def make_price_bars(n=250, start_price=150, seed=42):
    """Generate synthetic OHLCV bars."""
    np.random.seed(seed)
    prices = start_price + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 1.0)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").date

    bars = []
    for i in range(n):
        bars.append(MockBar(
            date=dates[i],
            open=float(prices[i] * 0.998),
            high=float(prices[i] * 1.01),
            low=float(prices[i] * 0.99),
            close=float(prices[i]),
            volume=int(1000000 + np.random.randint(-200000, 200000)),
        ))
    return bars


def make_fundamentals():
    """Generate mock fundamental data."""
    current = MockFundamental(
        items={
            "Revenues": 100e9,
            "NetIncomeLoss": 25e9,
            "Assets": 350e9,
            "Liabilities": 260e9,
            "StockholdersEquity": 90e9,
            "OperatingIncomeLoss": 30e9,
            "EarningsPerShareBasic": 6.0,
            "CommonStockSharesOutstanding": 15e9,
            "CashAndCashEquivalentsAtCarryingValue": 50e9,
            "LongTermDebt": 100e9,
        }
    )
    prior = MockFundamental(
        report_date=date(2024, 3, 31),
        items={
            "Revenues": 90e9,
            "NetIncomeLoss": 20e9,
            "StockholdersEquity": 80e9,
            "EarningsPerShareBasic": 5.0,
        }
    )
    return [current, prior]


# ============================================================================
# Macro Agent Tests
# ============================================================================

class TestMacroAgent:
    def test_bullish_macro(self):
        from marketmaster.agents.macro import MacroAgent
        agent = MacroAgent()
        plane = make_mock_plane()
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.agent == "macro"
        assert evidence.observations[0].startswith("MCEI score")
        assert evidence.scores["macro_alignment"] > 50
        assert len(evidence.bull_case) > 0
        assert evidence.confidence > 0

    def test_bearish_macro(self):
        from marketmaster.agents.macro import MacroAgent
        agent = MacroAgent()
        mcei = MockMceiResult(score=25.0, regime="BEAR", components={"liquidity": 20, "credit": 15})
        regime = MockRegimeResult(regime="BEAR", confidence=0.8)
        plane = make_mock_plane(mcei=mcei, regime=regime)
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.scores["macro_alignment"] < 50
        assert len(evidence.bear_case) > 0
        assert any("contractionary" in bc.lower() for bc in evidence.bear_case)

    def test_no_macro_data(self):
        from marketmaster.agents.macro import MacroAgent
        agent = MacroAgent()
        plane = make_mock_plane(mcei=None, regime=None)
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.data_quality == 0.0
        assert evidence.confidence == 0.0


# ============================================================================
# Fundamental Agent Tests
# ============================================================================

class TestFundamentalAgent:
    def test_strong_fundamentals(self):
        from marketmaster.agents.fundamental import FundamentalAgent
        agent = FundamentalAgent()
        plane = make_mock_plane(fundamentals=make_fundamentals())
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.agent == "fundamental"
        assert "ROE" in str(evidence.observations)
        assert evidence.scores.get("roe", 0) > 50  # Strong ROE
        assert evidence.scores.get("revenue_growth", 0) > 50  # Growing
        assert len(evidence.bull_case) > 0
        assert evidence.data_quality > 0.5

    def test_no_fundamentals(self):
        from marketmaster.agents.fundamental import FundamentalAgent
        agent = FundamentalAgent()
        plane = make_mock_plane(fundamentals=[])
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.data_quality == 0.0
        assert evidence.confidence == 0.0

    def test_high_debt_risk(self):
        from marketmaster.agents.fundamental import FundamentalAgent
        agent = FundamentalAgent()
        fund = make_fundamentals()
        fund[0].items["LongTermDebt"] = 250e9  # Very high debt
        plane = make_mock_plane(fundamentals=fund)
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert any("leverage" in r.lower() or "debt" in r.lower() for r in evidence.risks)


# ============================================================================
# Technical Agent Tests
# ============================================================================

class TestTechnicalAgent:
    def test_with_price_data(self):
        from marketmaster.agents.technical import TechnicalAgent
        agent = TechnicalAgent()
        bars = make_price_bars(250)
        plane = make_mock_plane(bars=bars)
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert evidence.agent == "technical"
        assert "close" in str(evidence.observations[0]).lower()
        assert "trend" in evidence.scores
        assert "rsi" in evidence.scores
        assert evidence.data_quality == 1.0
        assert evidence.confidence > 0

    def test_insufficient_data(self):
        from marketmaster.agents.technical import TechnicalAgent
        agent = TechnicalAgent()
        plane = make_mock_plane(bars=[])
        evidence = agent.analyze("AAPL", 1, date(2025, 6, 1), plane)

        assert "Insufficient" in evidence.observations[0]
        assert evidence.data_quality == 0.0


# ============================================================================
# Bull/Bear Debate Tests
# ============================================================================

class TestBullBearDebate:
    def _make_evidence(self, agent_name, scores, bull_case, bear_case, confidence=0.7, dq=0.8):
        from marketmaster.domain.models import DecisionEvidence
        return DecisionEvidence(
            agent=agent_name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            bull_case=bull_case,
            bear_case=bear_case,
            confidence=confidence,
            data_quality=dq,
        )

    def test_all_bullish(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()

        evidence = [
            self._make_evidence("technical", {"trend": 75, "rsi": 65}, ["Strong uptrend"], ["None"]),
            self._make_evidence("fundamental", {"roe": 80, "valuation": 70}, ["Strong ROE"], ["None"]),
        ]
        result = debate.run("AAPL", evidence)

        assert result.winner == "bull"
        assert result.bull_score > result.bear_score
        assert result.net_score > 0
        assert result.confidence > 0.3

    def test_all_bearish(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()

        evidence = [
            self._make_evidence("technical", {"trend": 25, "rsi": 20}, ["None"], ["Downtrend"]),
            self._make_evidence("fundamental", {"roe": 15, "valuation": 20}, ["None"], ["High P/E"]),
        ]
        result = debate.run("AAPL", evidence)

        assert result.winner == "bear"
        assert result.bear_score > result.bull_score
        assert result.net_score < 0

    def test_split_verdict(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()

        evidence = [
            self._make_evidence("technical", {"trend": 70, "rsi": 65}, ["Uptrend"], []),
            self._make_evidence("fundamental", {"roe": 25, "valuation": 20}, [], ["High P/E"]),
        ]
        result = debate.run("AAPL", evidence)

        # Should be split if scores are close
        assert result.winner in ("bull", "bear", "split")
        assert len(result.bull_arguments) > 0
        assert len(result.bear_arguments) > 0

    def test_contradiction_detection(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()

        evidence = [
            self._make_evidence("technical", {"trend": 80, "rsi": 75}, ["Strong revenue growth momentum"], []),
            self._make_evidence("fundamental", {"roe": 15, "growth": 10}, [], ["Revenue declining"]),
        ]
        result = debate.run("AAPL", evidence)

        # Should detect disagreement between agents
        assert len(result.contradictions) > 0 or len(result.cross_examination_notes) > 0

    def test_no_evidence(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()
        result = debate.run("AAPL", [])

        assert result.winner == "split"
        assert result.confidence == 0.0
        assert "No evidence" in result.summary

    def test_summary_generation(self):
        from marketmaster.agents.debate import BullBearDebate
        debate = BullBearDebate()

        evidence = [
            self._make_evidence("technical", {"trend": 75}, ["Strong uptrend"], []),
            self._make_evidence("macro", {"macro": 65}, ["Expansionary environment"], []),
        ]
        result = debate.run("AAPL", evidence)

        assert "AAPL" in result.summary
        assert "Verdict" in result.summary


# ============================================================================
# NLP Tests
# ============================================================================

class TestSentimentAnalysis:
    def test_positive_sentiment(self):
        from marketmaster.research.nlp import analyze_sentiment
        result = analyze_sentiment("The company reported strong revenue growth and beat expectations significantly.")
        assert result.score > 0
        assert result.label == "positive"
        assert "strong" in result.positive_terms
        assert "growth" in result.positive_terms
        assert "beat" in result.positive_terms

    def test_negative_sentiment(self):
        from marketmaster.research.nlp import analyze_sentiment
        result = analyze_sentiment("Revenue declined sharply amid deteriorating conditions and weak demand.")
        assert result.score < 0
        assert result.label == "negative"
        assert "declined" in result.negative_terms
        assert "deteriorating" in result.negative_terms

    def test_neutral_sentiment(self):
        from marketmaster.research.nlp import analyze_sentiment
        result = analyze_sentiment("The company reported quarterly earnings on Tuesday.")
        assert result.label == "neutral"
        assert abs(result.score) < 0.1

    def test_empty_text(self):
        from marketmaster.research.nlp import analyze_sentiment
        result = analyze_sentiment("")
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_forward_looking_detection(self):
        from marketmaster.research.nlp import analyze_sentiment
        text = "We expect revenue to grow next quarter and anticipate strong demand going forward."
        result = analyze_sentiment(text)
        assert result.forward_looking_count >= 3

    def test_hedging_detection(self):
        from marketmaster.research.nlp import analyze_sentiment
        text = "We estimate approximately $5 billion, subject to change, depending on market conditions."
        result = analyze_sentiment(text)
        assert result.hedging_count >= 3


class TestEntityExtraction:
    def test_ticker_extraction(self):
        from marketmaster.research.nlp import extract_entities
        result = extract_entities("AAPL and MSFT both reported earnings today.")
        assert "AAPL" in result.tickers
        assert "MSFT" in result.tickers

    def test_metric_extraction(self):
        from marketmaster.research.nlp import extract_entities
        result = extract_entities("Revenue grew 15% with EBITDA margin of 30%.")
        assert "revenue" in result.financial_metrics
        assert "EBITDA" in result.financial_metrics
        assert len(result.percentages) >= 2

    def test_dollar_amounts(self):
        from marketmaster.research.nlp import extract_entities
        result = extract_entities("Revenue was $5.2 billion, up from $4.8 billion.")
        assert len(result.numbers) >= 2


class TestTranscriptProcessing:
    def test_transcript_analysis(self):
        from marketmaster.research.nlp import process_transcript
        text = """
        Good morning. We are pleased to report strong revenue growth of 15% this quarter.
        Our operating margin improved significantly. We expect continued momentum going forward.
        However, we face some headwinds from rising costs and uncertain demand.

        Q&A:
        Can you discuss the competitive landscape? We are concerned about pricing pressure.
        """
        result = process_transcript(text)

        assert result.overall_sentiment.word_count > 0
        assert len(result.key_metrics) > 0
        assert len(result.forward_looking_statements) > 0
        assert len(result.risk_factors) > 0

    def test_prepared_vs_qa_sentiment(self):
        from marketmaster.research.nlp import process_transcript
        prepared = "We are delighted with strong growth and robust performance."
        qa = "We face declining demand and deteriorating conditions."

        result = process_transcript(prepared + "\n\nQ&A:\n" + qa)
        assert result.management_sentiment.score > 0
        assert result.qa_sentiment.score < 0
        assert result.tone_shift is not None
        assert result.tone_shift < 0  # Q&A more negative than prepared remarks


# ============================================================================
# Knowledge Retrieval Tests
# ============================================================================

class TestKnowledgeBase:
    def test_add_and_search(self):
        from marketmaster.research.knowledge import KnowledgeBase, Document

        kb = KnowledgeBase()
        kb.add_document(Document(
            doc_id="doc1",
            title="Apple Q1 Earnings Report",
            content="Apple reported strong revenue growth driven by iPhone sales and services revenue.",
            doc_type="news",
            symbols=["AAPL"],
        ))
        kb.add_document(Document(
            doc_id="doc2",
            title="Microsoft Azure Growth",
            content="Microsoft reported cloud revenue growth with Azure expanding market share.",
            doc_type="news",
            symbols=["MSFT"],
        ))

        assert kb.count() == 2

        results = kb.search("revenue growth")
        assert len(results) > 0
        assert results[0].score > 0

    def test_search_by_symbol(self):
        from marketmaster.research.knowledge import KnowledgeBase, Document

        kb = KnowledgeBase()
        kb.add_document(Document(
            doc_id="doc1", title="Apple News", content="Apple earnings",
            doc_type="news", symbols=["AAPL"],
        ))
        kb.add_document(Document(
            doc_id="doc2", title="Microsoft News", content="Microsoft earnings",
            doc_type="news", symbols=["MSFT"],
        ))

        results = kb.get_by_symbol("AAPL")
        assert len(results) == 1
        assert results[0].doc_id == "doc1"

    def test_filter_by_type(self):
        from marketmaster.research.knowledge import KnowledgeBase, Document

        kb = KnowledgeBase()
        kb.add_document(Document(
            doc_id="doc1", title="News", content="revenue growth",
            doc_type="news", symbols=["AAPL"],
        ))
        kb.add_document(Document(
            doc_id="doc2", title="Transcript", content="revenue growth discussion",
            doc_type="transcript", symbols=["AAPL"],
        ))

        results = kb.search("revenue growth", doc_type="transcript")
        assert len(results) == 1
        assert results[0].document.doc_type == "transcript"

    def test_empty_search(self):
        from marketmaster.research.knowledge import KnowledgeBase
        kb = KnowledgeBase()
        results = kb.search("anything")
        assert len(results) == 0


# ============================================================================
# Orchestrator Tests
# ============================================================================

class TestOrchestrator:
    def test_orchestrator_creation(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator
        orch = MarketMasterOrchestrator()
        assert len(orch.agents) >= 3  # At least macro, fundamental, technical
        assert orch.debate is not None

    def test_orchestrator_no_db(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator
        orch = MarketMasterOrchestrator()  # No db_session
        result = orch.analyze("AAPL")
        assert not result.data_available
        assert any("No database" in note for note in result.notes)

    def test_risk_gate(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator
        orch = MarketMasterOrchestrator()
        result = orch.check_risk(0.001, 0.001)
        assert not result.approved  # Live trading disabled by default
        assert "LIVE_TRADING_DISABLED" in result.reasons
