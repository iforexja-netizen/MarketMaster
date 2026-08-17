"""
Calibration Monitor — measures whether predicted confidence matches realized outcomes.

An overconfident trading system is dangerous. A model that says "65% win rate" but
actually wins 45% of the time will destroy capital over the long run. This module
detects that gap — the calibration gap — and quantifies it.

Calibration Concepts
─────────────────────
A prediction is *calibrated* when, among all the times the system predicted a
particular confidence level *p*, the observed win rate is also *p*. If we bin
predictions into confidence buckets (0–10%, 10–20%, …, 90–100%) and plot
predicted-vs-observed, a perfectly calibrated system lies on the 45° diagonal.

Brier Score Decomposition
─────────────────────────
The Brier score is the mean squared error of probabilistic predictions:

    BS = (1/N) Σ (f_i − o_i)²

where *f_i* is the predicted probability and *o_i* is the observed outcome (1 = win,
0 = loss). It decomposes into three terms:

    BS = REL  −  RES  +  UNC

  • REL (reliability)    — how far the calibration curve deviates from the diagonal.
    Lower is better. This is the calibration gap.
  • RES (resolution)     — how much the observed win rates in different bins differ
    from the base rate. Higher is better; it measures the system's ability to
    *separate* good predictions from bad ones.
  • UNC (uncertainty)    — the inherent variability of the outcomes.
    ō(1 − ō). Lower means outcomes are easier to predict.

Overconfidence Score
───────────────────
A weighted average of (predicted − observed) across all bins. Positive values mean
the system is systematically overconfident (claims higher win rates than it achieves).
Negative values mean underconfident.

Usage
─────
    monitor = CalibrationMonitor()

    # Record predictions as signals are generated
    monitor.record_prediction("sig_001", predicted_confidence=0.65,
                              predicted_score=72.0, strategy="trend_following",
                              regime="BULL")

    # Record outcomes once they're known
    monitor.record_outcome("sig_001", actual_win=True, pnl_pct=3.5)

    # Analyze
    result = monitor.compute_calibration(n_bins=10)
    print(result.brier_score)          # 0.18
    print(result.overconfidence_score) # +0.07  → overconfident
    print(result.recommendations)      # ['Strategy trend_following is overconfident at ...']

    # Per-strategy and per-regime breakdowns
    by_strat = monitor.get_calibration_by_strategy("trend_following")
    by_regime = monitor.get_calibration_by_regime("BULL")

    # Reliability diagram data for plotting
    diagram = monitor.get_reliability_diagram()
    # [{"predicted": 0.05, "observed": 0.04}, {"predicted": 0.15, "observed": 0.12}, ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# Internal record — one prediction + (optionally) its outcome
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _PredictionRecord:
    """Internal record linking a prediction to its realized outcome."""
    signal_id: str
    predicted_confidence: float        # 0.0–1.0 probability of a win
    predicted_score: float             # raw opportunity score (e.g. 0–100)
    strategy: str
    regime: str
    recorded_at: datetime
    # Outcome (filled in by record_outcome)
    actual_win: Optional[bool] = None
    pnl_pct: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None

    @property
    def has_outcome(self) -> bool:
        return self.actual_win is not None


# ════════════════════════════════════════════════════════════════════════
# CalibrationBin
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationBin:
    """
    A bin of predictions at similar confidence levels.

    Attributes:
        bin_low:            lower bound of the confidence range (inclusive).
        bin_high:           upper bound of the confidence range (exclusive,
                            except the last bin which is inclusive of 1.0).
        n_predictions:      total predictions in this bin that have outcomes.
        n_wins:             number of winning outcomes in this bin.
        observed_win_rate:  fraction of predictions that won (n_wins / n_predictions).
        predicted_confidence: average predicted confidence across predictions
                            in this bin.
        calibration_error:  absolute difference |observed − predicted|.
        samples:            the individual prediction records in this bin.
    """
    bin_low: float
    bin_high: float
    n_predictions: int = 0
    n_wins: int = 0
    observed_win_rate: float = 0.0
    predicted_confidence: float = 0.0
    calibration_error: float = 0.0
    samples: list = field(default_factory=list)  # list of _PredictionRecord

    @property
    def label(self) -> str:
        """Human-readable bin label, e.g. '60-70%'."""
        return f"{int(round(self.bin_low * 100))}-{int(round(self.bin_high * 100))}%"

    @property
    def is_overconfident(self) -> bool:
        """True when the system predicted a higher win rate than it achieved."""
        return self.predicted_confidence > self.observed_win_rate

    def to_dict(self) -> dict:
        return {
            "bin_low": self.bin_low,
            "bin_high": self.bin_high,
            "label": self.label,
            "n_predictions": self.n_predictions,
            "n_wins": self.n_wins,
            "observed_win_rate": self.observed_win_rate,
            "predicted_confidence": self.predicted_confidence,
            "calibration_error": self.calibration_error,
            "is_overconfident": self.is_overconfident,
        }


# ════════════════════════════════════════════════════════════════════════
# CalibrationResult
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """
    Full calibration analysis of a set of predictions.

    Attributes:
        bins:                   list of CalibrationBin, ordered by confidence.
        brier_score:            overall Brier score (lower is better).
        reliability:            reliability component of Brier (lower is better).
        resolution:             resolution component of Brier (higher is better).
        uncertainty:            uncertainty component of Brier (base-rate variance).
        overconfidence_score:   weighted (predicted − observed) across bins.
                                Positive → overconfident, negative → underconfident.
        most_overconfident_bin: the bin with the largest positive calibration error,
                                or None if every bin is well-calibrated / empty.
        recommendations:        actionable strings generated from the analysis.
        n_predictions:          total number of resolved predictions.
        n_wins:                 total wins among resolved predictions.
    """
    bins: list[CalibrationBin] = field(default_factory=list)
    brier_score: float = 0.0
    reliability: float = 0.0
    resolution: float = 0.0
    uncertainty: float = 0.0
    overconfidence_score: float = 0.0
    most_overconfident_bin: Optional[CalibrationBin] = None
    recommendations: list[str] = field(default_factory=list)
    n_predictions: int = 0
    n_wins: int = 0

    def to_dict(self) -> dict:
        return {
            "n_predictions": self.n_predictions,
            "n_wins": self.n_wins,
            "brier_score": self.brier_score,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "overconfidence_score": self.overconfidence_score,
            "most_overconfident_bin": (
                self.most_overconfident_bin.to_dict()
                if self.most_overconfident_bin
                else None
            ),
            "bins": [b.to_dict() for b in self.bins],
            "recommendations": self.recommendations,
        }


# ════════════════════════════════════════════════════════════════════════
# BrierScore — decomposition
# ════════════════════════════════════════════════════════════════════════

class BrierScore:
    """
    Compute and decompose the Brier score for binary probabilistic predictions.

    The Brier score measures the accuracy of probabilistic predictions. For a
    binary outcome it is the mean squared difference between predicted
    probabilities and actual outcomes (0 or 1).

    Decomposition (Murphy 1973 / Sanders decomposition):

        BS = REL − RES + UNC

    where:
        REL  = (1/N) Σ_k n_k (f̄_k − ō_k)²     — reliability (calibration gap)
        RES  = (1/N) Σ_k n_k (ō_k − ō)²        — resolution (separation power)
        UNC  = ō (1 − ō)                        — uncertainty (base-rate variance)

        N        = total number of predictions
        k        = bin index
        n_k      = number of predictions in bin k
        f̄_k     = mean predicted probability in bin k
        ō_k      = observed win rate in bin k
        ō        = overall base rate (total wins / N)

    A perfect Brier score is 0. A score near 0.25 means the predictions are no
    better than always guessing the base rate.
    """

    @staticmethod
    def compute(
        predictions: list[float],
        outcomes: list[bool],
        n_bins: int = 10,
    ) -> dict[str, float]:
        """
        Compute the Brier score and its decomposition.

        Args:
            predictions: predicted probabilities (0.0–1.0) for each event.
            outcomes:    actual outcomes (True/False or 1/0) for each event.
            n_bins:      number of equal-width bins for the decomposition.

        Returns:
            dict with keys ``brier``, ``reliability``, ``resolution``,
            ``uncertainty``.

        Raises:
            ValueError: if the two lists differ in length or are empty.
        """
        if len(predictions) != len(outcomes):
            raise ValueError(
                f"predictions ({len(predictions)}) and outcomes ({len(outcomes)}) "
                "must have the same length"
            )
        n = len(predictions)
        if n == 0:
            raise ValueError("Cannot compute Brier score for empty predictions")

        o = [1.0 if x else 0.0 for x in outcomes]

        # Overall Brier score
        brier = sum((f - oi) ** 2 for f, oi in zip(predictions, o)) / n

        # Overall base rate  ō
        o_bar = sum(o) / n

        # Uncertainty:  ō(1 − ō)
        uncertainty = o_bar * (1.0 - o_bar)

        # Bin predictions into equal-width [0,1] bins
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        reliability = 0.0
        resolution = 0.0

        for k in range(n_bins):
            low = bin_edges[k]
            high = bin_edges[k + 1]
            # Last bin includes 1.0
            if k == n_bins - 1:
                members = [
                    (f, oi)
                    for f, oi in zip(predictions, o)
                    if low <= f <= high
                ]
            else:
                members = [
                    (f, oi)
                    for f, oi in zip(predictions, o)
                    if low <= f < high
                ]

            n_k = len(members)
            if n_k == 0:
                continue

            f_bar_k = sum(f for f, _ in members) / n_k
            o_bar_k = sum(oi for _, oi in members) / n_k

            reliability += n_k * (f_bar_k - o_bar_k) ** 2
            resolution += n_k * (o_bar_k - o_bar) ** 2

        reliability /= n
        resolution /= n

        return {
            "brier": brier,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
        }


# ════════════════════════════════════════════════════════════════════════
# CalibrationMonitor
# ════════════════════════════════════════════════════════════════════════

class CalibrationMonitor:
    """
    Records predictions and outcomes, then computes calibration analysis.

    The monitor stores every prediction keyed by ``signal_id``. When the
    outcome arrives (via :meth:`record_outcome`) it attaches the realized
    result.  Calibration analysis only uses predictions that *have* outcomes —
    unresolved predictions are ignored.

    Usage::

        monitor = CalibrationMonitor()
        monitor.record_prediction("sig_001", 0.65, 72.0,
                                  strategy="trend_following", regime="BULL")
        monitor.record_outcome("sig_001", actual_win=True, pnl_pct=3.5)
        result = monitor.compute_calibration(n_bins=10)
    """

    def __init__(self) -> None:
        self._records: dict[str, _PredictionRecord] = {}

    # ── Recording ────────────────────────────────────────────────────────

    def record_prediction(
        self,
        signal_id: str,
        predicted_confidence: float,
        predicted_score: float = 0.0,
        strategy: str = "unknown",
        regime: str = "unknown",
    ) -> None:
        """
        Record a prediction before the outcome is known.

        Args:
            signal_id:             unique identifier for the prediction/signal.
            predicted_confidence:  estimated probability of a win (0.0–1.0).
            predicted_score:       raw opportunity score (any scale, stored for
                                    reference).
            strategy:              name of the strategy that produced the signal.
            regime:                market regime at the time of the prediction.
        """
        if not signal_id:
            raise ValueError("signal_id is required")
        if not (0.0 <= predicted_confidence <= 1.0):
            raise ValueError(
                f"predicted_confidence must be in [0, 1], got {predicted_confidence}"
            )

        self._records[signal_id] = _PredictionRecord(
            signal_id=signal_id,
            predicted_confidence=predicted_confidence,
            predicted_score=predicted_score,
            strategy=strategy,
            regime=regime,
            recorded_at=datetime.now(timezone.utc),
        )

    def record_outcome(
        self,
        signal_id: str,
        actual_win: bool,
        pnl_pct: float = 0.0,
    ) -> None:
        """
        Record the realized outcome for a previously-recorded prediction.

        Args:
            signal_id:  must match a prediction previously recorded.
            actual_win: whether the trade was a win (True) or loss (False).
            pnl_pct:    realised P&L percentage (e.g. 3.5 for +3.5 %).

        Raises:
            KeyError: if ``signal_id`` was never recorded as a prediction.
        """
        if signal_id not in self._records:
            raise KeyError(
                f"signal_id '{signal_id}' not found — call record_prediction first"
            )
        rec = self._records[signal_id]
        rec.actual_win = bool(actual_win)
        rec.pnl_pct = pnl_pct
        rec.outcome_recorded_at = datetime.now(timezone.utc)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolved_records(self) -> list[_PredictionRecord]:
        """Return only predictions that have a recorded outcome."""
        return [r for r in self._records.values() if r.has_outcome]

    @staticmethod
    def _bin_index(confidence: float, n_bins: int) -> int:
        """Map a confidence value to its bin index [0, n_bins-1]."""
        idx = int(math.floor(confidence * n_bins))
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        return idx

    def _compute_from_records(
        self,
        records: list[_PredictionRecord],
        n_bins: int,
        _skip_recommendations: bool = False,
    ) -> CalibrationResult:
        """Build a CalibrationResult from an arbitrary list of resolved records."""
        result = CalibrationResult(n_predictions=len(records))

        if not records:
            return result

        predictions = [r.predicted_confidence for r in records]
        outcomes = [r.actual_win for r in records]  # list[bool]
        n_total = len(records)
        n_wins = sum(1 for r in records if r.actual_win)
        result.n_predictions = n_total
        result.n_wins = n_wins

        # Brier decomposition
        decomp = BrierScore.compute(predictions, outcomes, n_bins=n_bins)
        result.brier_score = decomp["brier"]
        result.reliability = decomp["reliability"]
        result.resolution = decomp["resolution"]
        result.uncertainty = decomp["uncertainty"]

        # Build bins
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bins: list[CalibrationBin] = []
        for k in range(n_bins):
            low = bin_edges[k]
            high = bin_edges[k + 1]
            if k == n_bins - 1:
                members = [r for r in records if low <= r.predicted_confidence <= high]
            else:
                members = [r for r in records if low <= r.predicted_confidence < high]

            n_k = len(members)
            bin_obj = CalibrationBin(
                bin_low=low,
                bin_high=high,
                n_predictions=n_k,
                samples=members,
            )
            if n_k > 0:
                wins_k = sum(1 for r in members if r.actual_win)
                bin_obj.n_wins = wins_k
                bin_obj.observed_win_rate = wins_k / n_k
                bin_obj.predicted_confidence = sum(r.predicted_confidence for r in members) / n_k
                bin_obj.calibration_error = abs(bin_obj.observed_win_rate - bin_obj.predicted_confidence)
            bins.append(bin_obj)

        result.bins = bins

        # Overconfidence score: weighted average of (predicted − observed)
        # across non-empty bins, weighted by bin count.
        weighted_sum = 0.0
        total_weight = 0
        for b in bins:
            if b.n_predictions > 0:
                weighted_sum += (b.predicted_confidence - b.observed_win_rate) * b.n_predictions
                total_weight += b.n_predictions
        result.overconfidence_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Most overconfident bin (largest positive (predicted − observed) gap)
        best_bin: Optional[CalibrationBin] = None
        best_gap = 0.0
        for b in bins:
            if b.n_predictions == 0:
                continue
            gap = b.predicted_confidence - b.observed_win_rate
            if gap > best_gap:
                best_gap = gap
                best_bin = b
        result.most_overconfident_bin = best_bin

        # Recommendations (skip when computing internal sub-analyses to
        # avoid infinite recursion)
        if not _skip_recommendations:
            result.recommendations = self._generate_recommendations(bins, result)

        return result

    # ── Public analysis ──────────────────────────────────────────────────

    def compute_calibration(self, n_bins: int = 10) -> CalibrationResult:
        """
        Compute full calibration analysis across all resolved predictions.

        Args:
            n_bins: number of equal-width confidence bins (default 10).

        Returns:
            CalibrationResult with bins, Brier decomposition, overconfidence
            score, and recommendations.
        """
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        return self._compute_from_records(self._resolved_records(), n_bins)

    def get_calibration_by_strategy(self, strategy_name: str, n_bins: int = 10) -> CalibrationResult:
        """Per-strategy calibration analysis."""
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        records = [
            r for r in self._resolved_records() if r.strategy == strategy_name
        ]
        return self._compute_from_records(records, n_bins)

    def get_calibration_by_regime(self, regime: str, n_bins: int = 10) -> CalibrationResult:
        """Per-regime calibration analysis."""
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        records = [
            r for r in self._resolved_records() if r.regime == regime
        ]
        return self._compute_from_records(records, n_bins)

    def get_reliability_diagram(self, n_bins: int = 10) -> list[dict]:
        """
        Return data suitable for a reliability diagram plot.

        Each element is ``{"predicted": <avg confidence>, "observed": <win rate>}``
        for one bin.  Bins with zero predictions are skipped.

        Args:
            n_bins: number of equal-width confidence bins (default 10).

        Returns:
            list of ``{predicted, observed}`` dicts, ordered by bin.
        """
        result = self.compute_calibration(n_bins=n_bins)
        diagram: list[dict] = []
        for b in result.bins:
            if b.n_predictions > 0:
                diagram.append({
                    "predicted": b.predicted_confidence,
                    "observed": b.observed_win_rate,
                    "n_predictions": b.n_predictions,
                    "bin_low": b.bin_low,
                    "bin_high": b.bin_high,
                })
        return diagram

    def summary(self) -> dict:
        """
        Return a high-level summary dict for quick inspection / dashboards.
        """
        resolved = self._resolved_records()
        total_predictions = len(self._records)
        n_resolved = len(resolved)
        n_wins = sum(1 for r in resolved if r.actual_win)

        summary: dict = {
            "total_predictions": total_predictions,
            "resolved_predictions": n_resolved,
            "unresolved_predictions": total_predictions - n_resolved,
            "n_wins": n_wins,
            "overall_win_rate": (n_wins / n_resolved) if n_resolved > 0 else 0.0,
            "strategies_tracked": sorted({r.strategy for r in resolved}),
            "regimes_tracked": sorted({r.regime for r in resolved}),
        }

        if n_resolved > 0:
            result = self.compute_calibration()
            summary["brier_score"] = result.brier_score
            summary["reliability"] = result.reliability
            summary["resolution"] = result.resolution
            summary["uncertainty"] = result.uncertainty
            summary["overconfidence_score"] = result.overconfidence_score
            summary["most_overconfident_bin"] = (
                result.most_overconfident_bin.label
                if result.most_overconfident_bin
                else None
            )
            summary["recommendations"] = result.recommendations

        return summary

    def recommend(self) -> list[str]:
        """
        Return actionable recommendations based on current calibration.

        This is a convenience wrapper that calls :meth:`compute_calibration`
        and returns the ``recommendations`` field.
        """
        return self.compute_calibration().recommendations

    # ── Recommendation generation ─────────────────────────────────────────

    def _generate_recommendations(
        self,
        bins: list[CalibrationBin],
        result: CalibrationResult,
    ) -> list[str]:
        """Produce actionable recommendation strings from the analysis."""
        recs: list[str] = []

        # 1. Per-bin overconfidence warnings
        for b in bins:
            if b.n_predictions < 3:
                # not enough data to make a confident statement
                continue
            if b.predicted_confidence - b.observed_win_rate > 0.05:
                recs.append(
                    f"Overconfident at {b.label} confidence: predicted "
                    f"{b.predicted_confidence:.0%} win rate, observed "
                    f"{b.observed_win_rate:.0%} "
                    f"(gap {b.predicted_confidence - b.observed_win_rate:+.1%}, "
                    f"n={b.n_predictions})"
                )
            elif b.observed_win_rate - b.predicted_confidence > 0.05:
                recs.append(
                    f"Underconfident at {b.label} confidence: predicted "
                    f"{b.predicted_confidence:.0%} win rate, observed "
                    f"{b.observed_win_rate:.0%} "
                    f"(gap {b.predicted_confidence - b.observed_win_rate:+.1%}, "
                    f"n={b.n_predictions})"
                )

        # 2. Overall overconfidence assessment
        if result.overconfidence_score > 0.05:
            recs.append(
                f"System is systematically OVERCONFIDENT "
                f"(overconfidence score {result.overconfidence_score:+.3f}). "
                f"Consider dampening predicted confidences or tightening entry filters."
            )
        elif result.overconfidence_score < -0.05:
            recs.append(
                f"System is systematically UNDERCONFIDENT "
                f"(overconfidence score {result.overconfidence_score:+.3f}). "
                f"Predicted win rates are lower than observed — the system may be "
                f"too conservative and missing opportunities."
            )

        # 3. Brier score quality assessment
        if result.brier_score > 0.25:
            recs.append(
                f"Brier score {result.brier_score:.4f} is worse than the base-rate "
                f"baseline (0.25). Predictions add no value over always guessing "
                f"the base rate — review the scoring model."
            )
        elif result.brier_score > 0.20:
            recs.append(
                f"Brier score {result.brier_score:.4f} is near the base-rate "
                f"baseline. Prediction quality is marginal."
            )
        elif result.brier_score < 0.10 and result.n_predictions >= 20:
            recs.append(
                f"Brier score {result.brier_score:.4f} is excellent — predictions "
                f"are well-calibrated and discriminative."
            )

        # 4. Reliability vs resolution guidance
        if result.reliability > result.resolution and result.n_predictions >= 10:
            recs.append(
                f"Reliability ({result.reliability:.4f}) exceeds resolution "
                f"({result.resolution:.4f}) — the system is better at separating "
                f"good from bad predictions than at getting the absolute "
                f"probabilities right. Focus on calibration (e.g. Platt scaling "
                f"or temperature scaling)."
            )

        # 5. Low-resolution warning
        if (
            result.n_predictions >= 20
            and result.resolution < 0.01
            and result.uncertainty > 0.1
        ):
            recs.append(
                f"Resolution is very low ({result.resolution:.4f}) — the system "
                f"struggles to distinguish winning from losing predictions. "
                f"Consider adding discriminative features."
            )

        # 6. Per-strategy overconfidence — scan non-empty strategies
        # Compute per-strategy bins directly (skip recommendations to avoid
        # infinite recursion).
        strategy_set = sorted({r.strategy for r in self._resolved_records()})
        for strat in strategy_set:
            strat_records = [
                r for r in self._resolved_records() if r.strategy == strat
            ]
            strat_result = self._compute_from_records(
                strat_records, len(bins), _skip_recommendations=True
            )
            if strat_result.n_predictions < 5:
                continue
            for b in strat_result.bins:
                if b.n_predictions < 3:
                    continue
                gap = b.predicted_confidence - b.observed_win_rate
                if gap > 0.05:
                    recs.append(
                        f"Strategy {strat} is overconfident at {b.label} "
                        f"confidence: predicted {b.predicted_confidence:.0%} "
                        f"win rate, observed {b.observed_win_rate:.0%} "
                        f"(n={b.n_predictions})"
                    )
                    break  # one message per strategy

        # 7. Insufficient data
        if result.n_predictions < 20:
            recs.append(
                f"Only {result.n_predictions} resolved predictions — calibration "
                f"analysis may be unreliable. Collect more data before drawing "
                f"strong conclusions."
            )

        return recs


# ════════════════════════════════════════════════════════════════════════
# Convenience: empty result factory
# ════════════════════════════════════════════════════════════════════════

def empty_calibration_result() -> CalibrationResult:
    """Return an empty CalibrationResult (zero predictions)."""
    return CalibrationResult()
