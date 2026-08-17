"""
Sentiment Agent — Analyzes news, analyst sentiment, and social signals.

Domain: News items, analyst ratings, social sentiment, earnings call tone

The Sentiment Agent answers: What is the market's sentiment toward this
security? It aggregates qualitative signals into a quantitative sentiment score.

In Phase 3, this agent uses:
- News items from the news_items table (with pre-computed sentiment if available)
- Analyst consensus ratings (if available in signals/fundamentals)
- Earnings call transcript tone (if available)

The NLP processing layer (in research/nlp/) handles the actual text analysis.
This agent aggregates the results.
"""

from datetime import date, timedelta
from typing import Any, Optional

import numpy as np

from marketmaster.agents.base import SpecialistAgent
from marketmaster.domain.models import DecisionEvidence


class SentimentAgent(SpecialistAgent):
    """Aggregates sentiment signals from news, analysts, and social data."""

    def __init__(self, lookback_days: int = 30):
        super().__init__(
            name="sentiment",
            domain="sentiment",
            description="Analyzes news sentiment, analyst ratings, and social signals",
        )
        self.lookback_days = lookback_days

    def analyze(
        self,
        symbol: str,
        security_id: int,
        as_of: date,
        plane: Any,
    ) -> DecisionEvidence:
        evidence = self._make_evidence()
        lookback_start = as_of - timedelta(days=self.lookback_days)

        sentiment_scores = []

        # ── News Sentiment ───────────────────────────────────────────────────
        try:
            news = plane.get_news_items(security_id, start_date=lookback_start, end_date=as_of)
        except AttributeError:
            news = None
        except Exception:
            news = None

        if news:
            positive = 0
            negative = 0
            neutral = 0
            total_articles = len(news)

            for article in news:
                # Check if sentiment is pre-computed
                sent = None
                if hasattr(article, 'sentiment_score') and article.sentiment_score is not None:
                    sent = float(article.sentiment_score)
                elif hasattr(article, 'sentiment') and article.sentiment:
                    sent_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                    sent = sent_map.get(article.sentiment.lower(), 0.0) if isinstance(article.sentiment, str) else 0.0

                if sent is not None:
                    if sent > 0.2:
                        positive += 1
                    elif sent < -0.2:
                        negative += 1
                    else:
                        neutral += 1
                    sentiment_scores.append(sent)

            evidence.observations.append(
                f"News coverage: {total_articles} articles in {self.lookback_days}d "
                f"({positive} positive, {negative} negative, {neutral} neutral)"
            )

            if sentiment_scores:
                avg_sent = float(np.mean(sentiment_scores))
                evidence.scores["news_sentiment"] = self._safe_score(50 + avg_sent * 50)

                if avg_sent > 0.3:
                    evidence.bull_case.append(f"Strong positive news sentiment ({avg_sent:.2f})")
                elif avg_sent < -0.3:
                    evidence.bear_case.append(f"Negative news sentiment ({avg_sent:.2f})")
                    evidence.risks.append("Adverse news flow — monitor for fundamental deterioration")

            # News volume spike detection
            if total_articles > 20:
                evidence.observations.append("High news volume — elevated market attention")
                evidence.risks.append("High news volume may indicate catalyst or event risk")
        else:
            evidence.observations.append("No news data available for sentiment analysis")

        # ── Earnings Call Transcript Sentiment ────────────────────────────────
        try:
            transcripts = plane.get_transcripts(security_id, end_date=as_of)
        except AttributeError:
            transcripts = None
        except Exception:
            transcripts = None

        if transcripts:
            latest_transcript = transcripts[0]
            if hasattr(latest_transcript, 'sentiment_score') and latest_transcript.sentiment_score is not None:
                trans_sent = float(latest_transcript.sentiment_score)
                evidence.observations.append(
                    f"Latest earnings call sentiment: {trans_sent:.2f} "
                    f"({latest_transcript.call_date if hasattr(latest_transcript, 'call_date') else 'N/A'})"
                )
                evidence.scores["transcript_sentiment"] = self._safe_score(50 + trans_sent * 50)

                if trans_sent > 0.2:
                    evidence.bull_case.append("Positive tone in earnings call — management confidence")
                elif trans_sent < -0.2:
                    evidence.bear_case.append("Cautious tone in earnings call — management uncertainty")
        else:
            evidence.observations.append("No earnings call transcripts available")

        # ── Analyst Consensus (if available) ──────────────────────────────────
        try:
            signals = plane.get_signals(security_id, signal_type="analyst_rating", as_of_date=as_of)
        except (AttributeError, Exception):
            signals = None

        if signals:
            ratings = []
            for s in signals:
                if hasattr(s, 'score') and s.score is not None:
                    ratings.append(float(s.score))

            if ratings:
                avg_rating = float(np.mean(ratings))
                evidence.observations.append(
                    f"Analyst consensus: {avg_rating:.1f} (n={len(ratings)})"
                )
                evidence.scores["analyst_consensus"] = avg_rating

                if avg_rating > 70:
                    evidence.bull_case.append("Strong analyst consensus — bullish ratings")
                elif avg_rating < 30:
                    evidence.bear_case.append("Weak analyst consensus — bearish ratings")
        else:
            evidence.observations.append("No analyst consensus data available")

        # ── Social Sentiment (if available) ──────────────────────────────────
        try:
            social = plane.get_signals(security_id, signal_type="social_sentiment", as_of_date=as_of)
        except (AttributeError, Exception):
            social = None

        if social:
            social_scores = []
            for s in social:
                if hasattr(s, 'score') and s.score is not None:
                    social_scores.append(float(s.score))

            if social_scores:
                avg_social = float(np.mean(social_scores))
                evidence.observations.append(f"Social sentiment: {avg_social:.1f} (n={len(social_scores)})")
                evidence.scores["social_sentiment"] = avg_social

                if avg_social > 65:
                    evidence.bull_case.append("Positive social sentiment — retail optimism")
                elif avg_social < 35:
                    evidence.bear_case.append("Negative social sentiment — retail pessimism")

        # ── Data Quality ─────────────────────────────────────────────────────
        data_sources = sum([
            1 if news else 0,
            1 if transcripts else 0,
            1 if signals else 0,
            1 if social else 0,
        ])
        evidence.data_quality = min(1.0, data_sources / 4.0)

        # ── Confidence ───────────────────────────────────────────────────────
        if sentiment_scores:
            # Confidence is higher when sentiment is consistent
            sent_std = float(np.std(sentiment_scores))
            consistency = max(0.0, 1.0 - sent_std)
            evidence.confidence = min(0.8, consistency * (data_sources / 4.0))
        else:
            evidence.confidence = 0.1

        return evidence
