"""
NLP Layer — Sentiment analysis, entity extraction, and text processing.

This module provides the text analysis capabilities used by the Sentiment
Agent and the research pipeline:

1. Sentiment analysis: rule-based scoring of financial text
2. Entity extraction: identify companies, tickers, and financial terms
3. Transcript processing: segment earnings calls, detect forward-looking statements
4. Summarization: extract key points from financial documents

The sentiment analyzer uses a financial lexicon (Loughran-McDonald inspired)
rather than general-purpose NLP, because financial text has different
sentiment signals (e.g., "liability" is neutral in finance, negative in
general English).
"""

from dataclasses import dataclass, field
from typing import Optional

import re
from collections import Counter


# ============================================================================
# Financial Sentiment Lexicon
# ============================================================================

# Positive financial terms (Loughran-McDonald inspired)
POSITIVE_TERMS = {
    # Strong positive
    "outperform": 3, "beat": 3, "exceed": 3, "surpass": 3, "strong": 2,
    "growth": 2, "profit": 2, "gain": 2, "record": 2, "robust": 2,
    "momentum": 2, "expansion": 2, "improve": 2, "improved": 2, "improving": 2,
    "upgrade": 2, "upside": 2, "bullish": 3, "optimistic": 2, "confident": 2,
    "accelerate": 2, "accelerating": 2, "breakthrough": 3, "innovation": 2,
    "opportunity": 2, "opportunities": 2, "strong": 2, "surge": 3, "soar": 3,
    "rally": 2, "recovery": 2, "resilient": 2, "solid": 2, "stable": 1,
    # Moderate positive
    "increase": 1, "increased": 1, "increasing": 1, "higher": 1, "above": 1,
    "benefit": 1, "beneficial": 1, "positive": 1, "favorable": 1, "advantage": 1,
    "efficient": 1, "effective": 1, "competitive": 1, "demand": 1, "expanding": 1,
    "enhanced": 1, "advance": 1, "advancing": 1, "progress": 1, "success": 1,
}

# Negative financial terms
NEGATIVE_TERMS = {
    # Strong negative
    "miss": -3, "missed": -3, "underperform": -3, "plunge": -3, "crash": -3,
    "collapse": -3, "crisis": -3, "bankruptcy": -3, "insolvency": -3, "default": -3,
    "bearish": -3, "pessimistic": -2, "sell-off": -3, "selloff": -3, "slump": -2,
    "tumble": -3, "plummet": -3, "dive": -2, "spiral": -2, "freefall": -3,
    # Moderate negative
    "decline": -2, "declined": -2, "declining": -2, "decrease": -2, "decreased": -2,
    "fall": -2, "fell": -2, "falling": -2, "drop": -2, "dropped": -2, "lower": -1,
    "weak": -2, "weakness": -2, "loss": -2, "losses": -2, "deficit": -2,
    "risk": -1, "risks": -1, "concern": -2, "concerns": -2, "warning": -2,
    "caution": -2, "cautious": -2, "uncertain": -2, "uncertainty": -2,
    "pressure": -1, "headwind": -2, "headwinds": -2, "challenge": -1,
    "challenging": -2, "deteriorate": -3, "deteriorating": -3, "downgrade": -3,
    "below": -1, "shortfall": -2, "disappointing": -2, "disappointment": -2,
    "adverse": -2, "negative": -1, "unfavorable": -1, "recession": -3,
    "contraction": -2, "impairment": -2, "charge": -1, "litigation": -2,
    "investigation": -2, "lawsuit": -2, "fraud": -3, "restatement": -3,
}

# Forward-looking terms (important for transcripts)
FORWARD_LOOKING_TERMS = {
    "expect", "expects", "expected", "anticipate", "anticipates", "anticipated",
    "believe", "believes", "believed", "forecast", "project", "projects",
    "projected", "guidance", "outlook", "target", "targets", "goal", "goals",
    "plan", "plans", "planned", "intend", "intends", "aim", "aiming",
    "should", "will", "would", "could", "may", "might", "future",
    "going forward", "subsequent", "upcoming", "next quarter", "next year",
    "fiscal", "guidance", "revenue guidance", "earnings guidance",
}

# Cautious/hedging terms (reduce confidence in forward-looking statements)
HEDGING_TERMS = {
    "approximately", "approximately", "roughly", "about", "estimated",
    "estimate", "estimates", "subject to", "depending on", "may", "might",
    "could", "possibly", "potentially", "uncertain", "no assurance",
    "cannot guarantee", "may differ", "could vary", "subject to change",
}


# ============================================================================
# Sentiment Analysis
# ============================================================================

@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text."""
    score: float  # -1.0 to 1.0
    label: str  # "positive", "negative", "neutral"
    confidence: float  # 0-1
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    forward_looking_count: int = 0
    hedging_count: int = 0
    word_count: int = 0


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyze financial text sentiment using lexicon-based scoring.

    Returns a SentimentResult with score (-1 to 1), label, and confidence.

    This is intentionally simple and deterministic — no ML model needed.
    For production use, this could be enhanced with a fine-tuned BERT model,
    but the lexicon approach is transparent, fast, and bias-free.
    """
    if not text or not text.strip():
        return SentimentResult(score=0.0, label="neutral", confidence=0.0, word_count=0)

    # Tokenize (lowercase, split on non-word boundaries)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    word_count = len(words)

    if word_count == 0:
        return SentimentResult(score=0.0, label="neutral", confidence=0.0, word_count=0)

    # Score each word
    positive_hits = []
    negative_hits = []
    total_score = 0.0
    scored_words = 0

    for word in words:
        if word in POSITIVE_TERMS:
            total_score += POSITIVE_TERMS[word]
            positive_hits.append(word)
            scored_words += 1
        elif word in NEGATIVE_TERMS:
            total_score += NEGATIVE_TERMS[word]
            negative_hits.append(word)
            scored_words += 1

    # Also check for multi-word phrases
    text_lower = text.lower()
    for phrase in ["going forward", "next quarter", "next year", "revenue guidance",
                   "earnings guidance", "subject to", "depending on", "no assurance",
                   "cannot guarantee", "may differ", "could vary", "subject to change",
                   "sell-off", "sell off"]:
        count = text_lower.count(phrase)
        if count > 0:
            if phrase in ["sell-off", "sell off"]:
                total_score += count * NEGATIVE_TERMS.get("sell-off", -3)
                negative_hits.extend([phrase] * count)
                scored_words += count

    # Normalize to -1 to 1
    max_possible = max(scored_words * 3, 1)  # Max 3 per word
    score = total_score / max_possible if max_possible > 0 else 0.0
    score = max(-1.0, min(1.0, score))

    # Label
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # Confidence: higher when more words are scored
    coverage = scored_words / word_count if word_count > 0 else 0
    confidence = min(1.0, coverage * 5)  # Scale up since not all words are in lexicon

    # Forward-looking detection
    forward_count = 0
    for term in FORWARD_LOOKING_TERMS:
        forward_count += text_lower.count(term)

    # Hedging detection
    hedge_count = 0
    for term in HEDGING_TERMS:
        hedge_count += text_lower.count(term)

    return SentimentResult(
        score=round(score, 3),
        label=label,
        confidence=round(confidence, 3),
        positive_terms=list(dict.fromkeys(positive_hits)),  # unique, preserve order
        negative_terms=list(dict.fromkeys(negative_hits)),
        forward_looking_count=forward_count,
        hedging_count=hedge_count,
        word_count=word_count,
    )


# ============================================================================
# Entity Extraction
# ============================================================================

@dataclass
class EntityExtraction:
    """Result of entity extraction from text."""
    tickers: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    financial_metrics: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    percentages: list[str] = field(default_factory=list)


def extract_entities(text: str) -> EntityExtraction:
    """
    Extract financial entities from text.

    Identifies:
    - Tickers (e.g., "AAPL", "MSFT")
    - Financial metrics (e.g., "revenue", "EPS", "EBITDA")
    - Dollar amounts and percentages
    """
    result = EntityExtraction()

    # Tickers: all-caps words 1-5 chars (but filter common acronyms)
    ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
    common_acronyms = {"CEO", "CFO", "CTO", "COO", "IPO", "SEC", "GDP", "CPI",
                       "Fed", "FOMC", "ESG", "EBITDA", "EPS", "ROI", "ROE",
                       "YTD", "QTD", "MTD", "FY", "QA", "IR", "PR", "US",
                       "UK", "EU", "AP", "EDT", "EST", "PST", "AM", "PM"}
    for match in ticker_pattern.finditer(text):
        ticker = match.group(1)
        if ticker not in common_acronyms and len(ticker) >= 2:
            result.tickers.append(ticker)

    # Dollar amounts
    dollar_pattern = re.compile(r'\$[\d,]+\.?\d*[\s]*(?:billion|million|B|M|K|thousand)?', re.IGNORECASE)
    for match in dollar_pattern.finditer(text):
        result.numbers.append(match.group(0))

    # Percentages
    pct_pattern = re.compile(r'\d+\.?\d*\s*%')
    for match in pct_pattern.finditer(text):
        result.percentages.append(match.group(0))

    # Financial metrics
    metrics = ["revenue", "earnings", "EBITDA", "EPS", "margin", "cash flow",
               "operating income", "net income", "gross profit", "book value",
               "debt", "leverage", "P/E", "P/B", "ROE", "ROA", "ROI",
               "free cash flow", "FCF", "working capital", "capex"]
    text_lower = text.lower()
    for metric in metrics:
        if metric.lower() in text_lower:
            result.financial_metrics.append(metric)

    # Unique
    result.tickers = list(dict.fromkeys(result.tickers))
    result.financial_metrics = list(dict.fromkeys(result.financial_metrics))

    return result


# ============================================================================
# Transcript Processing
# ============================================================================

@dataclass
class TranscriptAnalysis:
    """Analysis of an earnings call transcript."""
    overall_sentiment: SentimentResult
    management_sentiment: SentimentResult
    qa_sentiment: Optional[SentimentResult] = None
    key_metrics: list[str] = field(default_factory=list)
    forward_looking_statements: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    tone_shift: Optional[float] = None  # Change from prepared remarks to Q&A


def process_transcript(
    transcript_text: str,
    prepared_remarks: Optional[str] = None,
    qa_section: Optional[str] = None,
) -> TranscriptAnalysis:
    """
    Process an earnings call transcript for sentiment and key information.

    Args:
        transcript_text: Full transcript text
        prepared_remarks: Optional separated prepared remarks section
        qa_section: Optional separated Q&A section

    Returns TranscriptAnalysis with sentiment and extracted information.
    """
    # Overall sentiment
    overall_sentiment = analyze_sentiment(transcript_text)

    # Management (prepared remarks) sentiment
    if prepared_remarks:
        management_sentiment = analyze_sentiment(prepared_remarks)
    else:
        # Try to split: assume Q&A starts with "Q&A" or "Questions and Answers"
        split_patterns = [r"Q\s*&\s*A", r"Questions?\s+and\s+Answers?",
                          r"Question-and-Answer", r"\bQ\b\s*:\s*"]
        split_text = transcript_text
        for pattern in split_patterns:
            match = re.search(pattern, transcript_text, re.IGNORECASE)
            if match:
                prepared_remarks = transcript_text[:match.start()]
                qa_section = transcript_text[match.start():]
                break

        if prepared_remarks:
            management_sentiment = analyze_sentiment(prepared_remarks)
        else:
            management_sentiment = overall_sentiment

    # Q&A sentiment
    qa_sentiment = None
    if qa_section:
        qa_sentiment = analyze_sentiment(qa_section)

    # Tone shift: difference between prepared remarks and Q&A
    tone_shift = None
    if qa_sentiment:
        tone_shift = qa_sentiment.score - management_sentiment.score

    # Extract entities
    entities = extract_entities(transcript_text)
    key_metrics = entities.financial_metrics

    # Extract forward-looking statements (sentences with forward-looking terms)
    sentences = re.split(r'[.!?]+', transcript_text)
    forward_statements = []
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if any(term.lower() in sentence_lower for term in FORWARD_LOOKING_TERMS):
            if len(sentence.strip()) > 20:  # Skip very short sentences
                forward_statements.append(sentence.strip()[:200])  # Cap length

    # Extract risk factors (sentences with negative terms)
    risk_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        risk_terms = ["risk", "concern", "uncertain", "challenge", "headwind",
                      "adverse", "deteriorate", "pressure", "could decline",
                      "may decline", "warning"]
        if any(term in sentence_lower for term in risk_terms):
            if len(sentence.strip()) > 20:
                risk_sentences.append(sentence.strip()[:200])

    return TranscriptAnalysis(
        overall_sentiment=overall_sentiment,
        management_sentiment=management_sentiment,
        qa_sentiment=qa_sentiment,
        key_metrics=key_metrics,
        forward_looking_statements=forward_statements[:10],  # Top 10
        risk_factors=risk_sentences[:5],  # Top 5
        tone_shift=tone_shift,
    )


# ============================================================================
# Batch Sentiment
# ============================================================================

def batch_sentiment(texts: list[str]) -> list[SentimentResult]:
    """Analyze sentiment for multiple texts."""
    return [analyze_sentiment(text) for text in texts]


def aggregate_sentiment(results: list[SentimentResult]) -> SentimentResult:
    """
    Aggregate multiple sentiment results into one.

    Used when analyzing multiple news articles about the same security.
    """
    if not results:
        return SentimentResult(score=0.0, label="neutral", confidence=0.0, word_count=0)

    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores)

    all_positive = []
    all_negative = []
    for r in results:
        all_positive.extend(r.positive_terms)
        all_negative.extend(r.negative_terms)

    # Most common terms
    top_positive = [term for term, _ in Counter(all_positive).most_common(10)]
    top_negative = [term for term, _ in Counter(all_negative).most_common(10)]

    # Confidence: based on consistency and coverage
    score_std = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
    consistency = max(0.0, 1.0 - score_std)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    if avg_score > 0.1:
        label = "positive"
    elif avg_score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        score=round(avg_score, 3),
        label=label,
        confidence=round(consistency * avg_confidence, 3),
        positive_terms=top_positive,
        negative_terms=top_negative,
        forward_looking_count=sum(r.forward_looking_count for r in results),
        hedging_count=sum(r.hedging_count for r in results),
        word_count=sum(r.word_count for r in results),
    )
