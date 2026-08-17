# MarketMaster V1 Technical Architecture

## Layers
1. Data: equities, ETFs, options, fundamentals, SEC filings, macro, news/transcripts, account data.
2. Feature Store: returns, volatility, ATR, RSI, ADX, moving averages, relative strength, valuation, profitability, leverage, liquidity, MCEI, IV/Greeks/skew/term structure.
3. Intelligence Engines: Fundamental, Technical, Macro/MCEI, Options, Sentiment/Research, Regime.
4. Agent Layer: CEO/Orchestrator plus specialist agents.
5. Decision: opportunity score -> strategy -> expected value -> risk sizing -> portfolio constraints.
6. Execution: paper broker first; live execution behind explicit flags and hard gates.
7. Learning: every signal, decision, order, fill and outcome is attributable.

## Agents
CEO/Orchestrator, Macro, Fundamental, Technical, Quant, Options, Sentiment/Research, Regime, Strategy, Risk, Portfolio, Execution, Audit, Learning.

## Agent contract
Every agent returns structured evidence containing agent, timestamp, observations, scores, bull case, bear case, risks, data quality, confidence, and recommended actions.

LLMs may generate hypotheses/explanations. They are not the authoritative source for numerical market data.

## Decision pipeline
Ingest -> validate -> features -> regime -> candidates -> specialist analysis -> bull/bear debate -> quant validation -> strategy -> expected value -> risk gate -> portfolio constraints -> human approval in V1 -> paper execution -> monitoring -> attribution -> model evaluation.

## Initial opportunity score
Fundamental quality 15%
Valuation 10%
Technical structure 15%
Momentum/relative strength 10%
Macro alignment 10%
Options opportunity 10%
Catalyst 5%
Sentiment 5%
Liquidity 5%
Risk/reward 15%

Weights are configurable and later evaluated out-of-sample.

## MCEI
Initial components:
- broad money growth
- bank credit growth
- commercial & industrial lending
- consumer credit
- Fed balance sheet
- fiscal/Treasury liquidity proxies
- yield curve
- real yields
- credit spreads
- financial conditions

Transform components into historical percentile/z-score measures with signs aligned so higher means more expansionary. MCEI is a weighted normalized 0-100 composite. Regime thresholds must be validated with historical walk-forward testing rather than treated as permanent truths.

## Regimes
STRONG_BULL, BULL, TRANSITION_BULL, NEUTRAL, TRANSITION_BEAR, BEAR, CRISIS, RECOVERY.

## Strategy registry
Quality Compounder, Value, Momentum, Breakout, Pullback, Mean Reversion, Relative Strength, Sector Rotation, Covered Call, Cash-Secured Put, Vertical Spread, Iron Condor, Iron Butterfly, Broken-Wing Butterfly, Calendar, Diagonal.

Each strategy declares universe, prerequisites, entry, exit, invalidation, sizing, costs, allowed regimes and prohibited conditions.

## Risk gate
Hard controls:
- max position risk
- sector/correlation exposure
- beta/portfolio volatility
- daily loss and drawdown
- liquidity
- options spread/liquidity
- earnings/event restrictions
- stale-data rejection
- broker/account permissions

The deterministic Risk Gate has final authority.

## Backtesting
Prevent look-ahead bias, use point-in-time data when available, model commissions/slippage/bid-ask, corporate actions, historical universes, train/validation/test splits and walk-forward validation. Report CAGR, Sharpe, Sortino, max drawdown, win rate, expectancy, profit factor, turnover, exposure and tail loss.

## Dashboard
Market Overview, Regime, MCEI, Screener, Opportunity Radar, Stock Detail, Options Lab, Strategy Lab, Portfolio, Risk, AI Research, Backtests, Trade Journal, Model Performance, Audit.

## V1 acceptance
MarketMaster can ingest a US equity universe, calculate core features and MCEI, classify regime, rank opportunities, generate structured research, propose strategies, calculate risk, reject invalid trades, backtest without look-ahead, paper trade, explain decisions and measure signal value.
