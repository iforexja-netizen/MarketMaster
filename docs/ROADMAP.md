# MarketMaster Roadmap

## Phase 0 — Foundation ✅ COMPLETE
## Phase 1 — Data Plane ✅ COMPLETE
## Phase 2 — MCEI & Quant Engines ✅ COMPLETE
## Phase 3 — Research Plane ✅ COMPLETE
## Phase 4 — Strategy Plane ✅ COMPLETE
## Phase 5 — Risk + Paper Trading ✅ COMPLETE

## Phase 6 — Learning System ✅ COMPLETE
### Signal Attribution Framework
- [x] Links every trade signal to its realized outcome (entry → exit P&L)
- [x] Tracks agent evidence at signal time (which agents contributed to the decision)
- [x] Records debate score and winner at entry
- [x] Computes R-multiple (pnl / initial risk) for every trade
- [x] Exit reason tracking (stop_loss, take_profit, signal_exit, time_exit, manual)
- [x] Regime at entry and exit
- [x] Per-strategy performance stats (win rate, avg R, expectancy, profit factor, Sharpe)
- [x] Per-agent contribution stats (avg score on wins vs losses, correlation with P&L)
- [x] Per-regime performance stats
- [x] Summary with overall metrics

### Calibration Monitoring
- [x] 10-bin calibration analysis (predicted vs observed win rates by confidence bucket)
- [x] Brier score decomposition (reliability, resolution, uncertainty)
- [x] Overconfidence detection (predicted > observed consistently)
- [x] Per-strategy and per-regime calibration breakdowns
- [x] Reliability diagram data for plotting
- [x] Actionable recommendations (e.g. "Strategy X is overconfident at 60-70% confidence")
- [x] BrierScore class with full decomposition

### Model Registry and Versioning
- [x] Versioned model configurations (strategy, regime engine, MCEI, scoring, risk)
- [x] Parameter hashing for deduplication
- [x] Model lifecycle: experimental → candidate → production → deprecated → retired
- [x] Version activation (deactivates previous production version)
- [x] Model lineage tracking (parent → child version chain)
- [x] Version comparison (parameter diffs, new/removed params, metric changes)
- [x] Performance metrics per version
- [x] Summary and export

### Drift Detection
- [x] Feature drift via Population Stability Index (PSI)
- [x] Performance drift via CUSUM (cumulative sum of deviations)
- [x] Regime stability monitoring (change rate detection)
- [x] Volatility regime drift detection
- [x] FeatureBaseline class with histogram and PSI computation
- [x] Severity classification (none → low → moderate → high → severe)
- [x] Full drift report with alerts and recommendations
- [x] Actionable recommendations per alert

### Strategy Ranking
- [x] Ranks strategies by realized performance (not backtests)
- [x] 7 ranking metrics: expectancy, R-multiple, win rate, profit factor, Sharpe, consistency, composite
- [x] Composite score (weighted combination, 0-100)
- [x] Edge persistence measurement (first-half vs second-half comparison)
- [x] Regime-specific performance breakdown
- [x] Allocation action recommendations (increase, maintain, decrease, pause, investigate)
- [x] Actionable recommendations for portfolio rebalancing
- [x] Human-readable summary

### API Endpoints (Phase 6)
- [x] POST /learning/attribution/entry — Record signal entry
- [x] POST /learning/attribution/exit — Record trade exit
- [x] GET /learning/attribution/strategies — Strategy performance stats
- [x] GET /learning/attribution/agents — Agent contribution stats
- [x] POST /learning/calibration — Compute calibration analysis
- [x] GET /learning/calibration/recommendations — Calibration recommendations
- [x] POST /learning/registry/register — Register model version
- [x] POST /learning/registry/{id}/activate — Activate version
- [x] GET /learning/registry/models — List all versions
- [x] GET /learning/registry/active — List active versions
- [x] GET /learning/registry/compare — Compare two versions
- [x] POST /learning/drift/check — Check for drift
- [x] GET /learning/drift/alerts — Get drift alerts
- [x] GET /learning/ranking — Strategy ranking report

### Testing
- [x] 263 tests passing across 10 test files (60 new Phase 6 tests)
  - Signal attribution (8 tests): creation, R-multiple, entry/exit tracking, strategy stats
  - Calibration (6 tests): prediction recording, well-calibrated, overconfident, Brier decomposition
  - Model registry (13 tests): register, activate, deprecate, lineage, compare, retire, summary
  - Drift detection (13 tests): PSI baseline, feature drift, performance drift, regime stability, volatility
  - Strategy ranking (13 tests): ranking, allocation actions, edge persistence, recommendations

## Phase 7 — Controlled Live Trading (NEXT)
- [ ] Live trading flag + permission validation
- [ ] Hard loss limits + kill switch integration
- [ ] Position reconciliation
- [ ] Independent audit capability
- [ ] Gradual capital deployment

## Current Status
- **87 Python files** | **263 tests passing** | **0 failures**
- **Phase 0 (Foundation):** ✅ Complete
- **Phase 1 (Data Plane):** ✅ Complete
- **Phase 2 (MCEI + Quant Engines):** ✅ Complete
- **Phase 3 (Research Plane):** ✅ Complete
- **Phase 4 (Strategy Plane):** ✅ Complete
- **Phase 5 (Risk + Paper Trading):** ✅ Complete
- **Phase 6 (Learning System):** ✅ Complete
- **Phase 7 (Controlled Live Trading):** Next up

## Full Pipeline: Data → MCEI → Quant → Regime → Agents → Debate → Strategies →
                 Screener → Portfolio → Risk Gate → Broker → Positions → Audit Trail →
                 Attribution → Calibration → Drift Detection → Strategy Ranking

**The system now learns from every trade. It knows which strategies work, which agents add edge,
which predictions were overconfident, and when the market is drifting from what models expect.**
