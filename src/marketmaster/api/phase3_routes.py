"""
MarketMaster API Routes — Phase 3: Research Plane

Endpoints for specialist agent analysis, bull/bear debate, NLP sentiment,
and knowledge retrieval.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from marketmaster.db.session import get_db
from marketmaster.data.plane import DataPlane

phase3_router = APIRouter()


# ============================================================================
# Agent Analysis
# ============================================================================

@phase3_router.get("/analysis/{symbol}")
def analyze_symbol(
    symbol: str,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Full multi-agent analysis with bull/bear debate.

    Dispatches to all specialist agents (macro, fundamental, technical,
    options, sentiment), runs the bull/bear debate, and returns the
    complete analysis with evidence, scores, and debate verdict.
    """
    from marketmaster.agents.orchestrator import MarketMasterOrchestrator

    if as_of is None:
        as_of = date.today()

    orch = MarketMasterOrchestrator(db_session=db)
    return orch.get_full_analysis(symbol, as_of)


@phase3_router.get("/analysis/{symbol}/agents")
def get_agent_evidence(
    symbol: str,
    agent_name: Optional[str] = None,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Get evidence from specialist agents for a security.

    If agent_name is provided, returns only that agent's evidence.
    Otherwise returns all agent evidence.
    """
    from marketmaster.agents.orchestrator import MarketMasterOrchestrator

    if as_of is None:
        as_of = date.today()

    orch = MarketMasterOrchestrator(db_session=db)
    result = orch.analyze(symbol, as_of)

    if not result.data_available:
        raise HTTPException(status_code=404, detail=result.notes[-1])

    evidence_list = result.evidence
    if agent_name:
        evidence_list = [e for e in evidence_list if e.agent == agent_name]
        if not evidence_list:
            raise HTTPException(status_code=404, detail=f"Agent not found or no data: {agent_name}")

    return {
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "agents": [
            {
                "agent": ev.agent,
                "observations": ev.observations,
                "scores": ev.scores,
                "bull_case": ev.bull_case,
                "bear_case": ev.bear_case,
                "risks": ev.risks,
                "data_quality": ev.data_quality,
                "confidence": ev.confidence,
                "recommended_actions": ev.recommended_actions,
            }
            for ev in evidence_list
        ],
    }


@phase3_router.get("/analysis/{symbol}/debate")
def get_debate(
    symbol: str,
    as_of: Optional[date] = None,
    db: Session = Depends(get_db),
):
    """
    Get the bull/bear debate result for a security.

    Returns the structured debate: bull score, bear score, winner,
    confidence, surviving arguments, contradictions, and key risks.
    """
    from marketmaster.agents.orchestrator import MarketMasterOrchestrator

    if as_of is None:
        as_of = date.today()

    orch = MarketMasterOrchestrator(db_session=db)
    result = orch.analyze(symbol, as_of)

    if not result.debate:
        raise HTTPException(
            status_code=422,
            detail="Unable to run debate — insufficient agent evidence",
        )

    debate = result.debate
    return {
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "bull_score": debate.bull_score,
        "bear_score": debate.bear_score,
        "net_score": debate.net_score,
        "winner": debate.winner,
        "confidence": debate.confidence,
        "bull_arguments": [
            {
                "agent": a.agent,
                "argument": a.argument,
                "strength": a.evidence_strength,
                "survives": a.survives_cross_examination,
            }
            for a in debate.bull_arguments
        ],
        "bear_arguments": [
            {
                "agent": a.agent,
                "argument": a.argument,
                "strength": a.evidence_strength,
                "survives": a.survives_cross_examination,
            }
            for a in debate.bear_arguments
        ],
        "cross_examination": debate.cross_examination_notes,
        "contradictions": debate.contradictions,
        "key_risks": debate.key_risks,
        "summary": debate.summary,
    }


# ============================================================================
# NLP — Sentiment Analysis
# ============================================================================

class SentimentRequest(BaseModel):
    text: str


@phase3_router.post("/nlp/sentiment")
def analyze_text_sentiment(req: SentimentRequest):
    """
    Analyze sentiment of financial text.

    Uses a financial lexicon (Loughran-McDonald inspired) to score
    text from -1 (very negative) to +1 (very positive).

    Also detects forward-looking statements and hedging language.
    """
    from marketmaster.research.nlp import analyze_sentiment

    result = analyze_sentiment(req.text)
    return {
        "score": result.score,
        "label": result.label,
        "confidence": result.confidence,
        "positive_terms": result.positive_terms,
        "negative_terms": result.negative_terms,
        "forward_looking_count": result.forward_looking_count,
        "hedging_count": result.hedging_count,
        "word_count": result.word_count,
    }


@phase3_router.post("/nlp/entities")
def extract_entities_endpoint(req: SentimentRequest):
    """
    Extract financial entities from text (tickers, metrics, numbers).
    """
    from marketmaster.research.nlp import extract_entities

    result = extract_entities(req.text)
    return {
        "tickers": result.tickers,
        "financial_metrics": result.financial_metrics,
        "numbers": result.numbers,
        "percentages": result.percentages,
    }


@phase3_router.post("/nlp/transcript")
def analyze_transcript(req: SentimentRequest):
    """
    Process an earnings call transcript for sentiment and key information.

    Returns overall sentiment, management vs Q&A tone shift, forward-looking
    statements, and risk factors.
    """
    from marketmaster.research.nlp import process_transcript

    result = process_transcript(req.text)
    return {
        "overall_sentiment": {
            "score": result.overall_sentiment.score,
            "label": result.overall_sentiment.label,
            "confidence": result.overall_sentiment.confidence,
        },
        "management_sentiment": {
            "score": result.management_sentiment.score,
            "label": result.management_sentiment.label,
            "confidence": result.management_sentiment.confidence,
        },
        "qa_sentiment": {
            "score": result.qa_sentiment.score,
            "label": result.qa_sentiment.label,
            "confidence": result.qa_sentiment.confidence,
        } if result.qa_sentiment else None,
        "tone_shift": result.tone_shift,
        "key_metrics": result.key_metrics,
        "forward_looking_statements": result.forward_looking_statements,
        "risk_factors": result.risk_factors,
    }


# ============================================================================
# Knowledge Retrieval
# ============================================================================

class DocumentRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    doc_type: str
    symbols: list[str] = []
    source: Optional[str] = None


@phase3_router.post("/knowledge/documents")
def add_document(req: DocumentRequest, db: Session = Depends(get_db)):
    """
    Add a document to the knowledge base for retrieval.
    """
    from marketmaster.research.knowledge import KnowledgeBase, Document
    from datetime import date

    # In production, this would persist to the database
    # For now, we use an in-memory knowledge base per request
    # (A persistent store would be implemented in Phase 4)
    kb = KnowledgeBase()
    doc = Document(
        doc_id=req.doc_id,
        title=req.title,
        content=req.content,
        doc_type=req.doc_type,
        symbols=req.symbols,
        source=req.source,
    )
    kb.add_document(doc)
    return {
        "status": "indexed",
        "doc_id": req.doc_id,
        "doc_type": req.doc_type,
        "total_documents": kb.count(),
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    doc_type: Optional[str] = None


@phase3_router.post("/knowledge/search")
def search_documents(req: SearchRequest):
    """
    Search the knowledge base for relevant documents.

    Uses TF-IDF cosine similarity for relevance ranking.
    """
    from marketmaster.research.knowledge import KnowledgeBase

    # In production, this would query a persistent store
    kb = KnowledgeBase()
    results = kb.search(req.query, top_k=req.top_k, doc_type=req.doc_type)
    return {
        "query": req.query,
        "results": [
            {
                "doc_id": r.document.doc_id,
                "title": r.document.title,
                "doc_type": r.document.doc_type,
                "score": r.score,
                "snippet": r.snippet,
            }
            for r in results
        ],
    }


# ============================================================================
# Agent Registry
# ============================================================================

@phase3_router.get("/agents")
def list_agents():
    """
    List all available specialist agents and their descriptions.
    """
    return {
        "agents": [
            {"name": "macro", "domain": "macro", "description": "MCEI, liquidity, rates, yield curve, regime"},
            {"name": "fundamental", "domain": "fundamental", "description": "Financials, valuation, growth, quality"},
            {"name": "technical", "domain": "technical", "description": "Price action, trends, momentum, volume"},
            {"name": "options", "domain": "options", "description": "IV, put/call ratio, skew, gamma exposure"},
            {"name": "sentiment", "domain": "sentiment", "description": "News, analyst ratings, social sentiment"},
        ]
    }
