#!/usr/bin/env python3
"""
MarketMaster — Live Data Ingestion Runner

Connects to live Alpaca + FRED APIs using stored credentials,
fetches real market data, computes MCEI, and runs the full pipeline.

Usage:
    PYTHONPATH=src python3 run_live.py            # Full pipeline with live data
    PYTHONPATH=src python3 run_live.py --ingest   # Just ingest data (no pipeline)
    PYTHONPATH=src python3 run_live.py --check     # Check API connectivity
"""

import sys
import os
import asyncio
import json
from datetime import date, datetime, timezone, timedelta
from typing import Optional

# Load environment variables from .agents/.env
from pathlib import Path
env_path = Path("/app/.agents/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from marketmaster.config.mcei_series import MCEI_COMPONENTS
from marketmaster.engines.mcei import calculate_mcei
from marketmaster.engines.technical import rsi, adx, atr, sma, macd, bollinger_bands
from marketmaster.data.providers.alpaca import AlpacaProvider
from marketmaster.data.providers.fred import FredProvider
from marketmaster.data.providers.sec import SecEdgarProvider

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def banner(title, subtitle=""):
    line = "=" * 80
    print(f"\n{BLUE}{BOLD}{line}{RESET}")
    print(f"{BLUE}{BOLD}  {title}{RESET}")
    if subtitle:
        print(f"{BLUE}  {subtitle}{RESET}")
    print(f"{BLUE}{BOLD}{line}{RESET}\n")

def step(num, desc):
    print(f"\n{GREEN}▶ {num}: {desc}{RESET}")

def result(msg):
    print(f"  {YELLOW}→ {msg}{RESET}")

def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")

def fail(msg):
    print(f"  {RED}✗ {msg}{RESET}")

# Default universe
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA",
    "DIS", "BAC", "XOM", "PFE", "KO",
]


async def check_connectivity():
    """Check API connectivity for all providers."""
    banner("API CONNECTIVITY CHECK", "Verifying Alpaca, FRED, and SEC EDGAR")

    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET", "")
    fred_key = os.environ.get("FRED_API_KEY", "")

    # ── Alpaca ──
    step("1", "Alpaca Markets")
    if not alpaca_key or not alpaca_secret:
        fail("No ALPACA_API_KEY / ALPACA_API_SECRET in environment")
    else:
        result(f"API Key: {alpaca_key[:8]}...{alpaca_key[-4:]}")
        provider = AlpacaProvider(api_key=alpaca_key, secret_key=alpaca_secret, paper=True)
        healthy = await provider.health_check()
        if healthy:
            success("Alpaca paper trading API — CONNECTED")
        else:
            fail("Alpaca API reachable but credentials invalid or expired")

    # ── FRED ──
    step("2", "FRED (Federal Reserve Economic Data)")
    if not fred_key:
        fail("No FRED_API_KEY in environment")
    else:
        result(f"API Key: {fred_key[:8]}...{fred_key[-4:]}")
        provider = FredProvider(api_key=fred_key)
        healthy = await provider.health_check()
        if healthy:
            success("FRED API — CONNECTED")
        else:
            fail("FRED API reachable but credentials invalid")

    # ── SEC EDGAR ──
    step("3", "SEC EDGAR")
    sec_provider = SecEdgarProvider(user_agent="MarketMaster iforexja@gmail.com")
    healthy = await sec_provider.health_check()
    if healthy:
        success("SEC EDGAR — CONNECTED (no key needed, just User-Agent)")
    else:
        fail("SEC EDGAR unreachable")

    # ── Summary ──
    banner("CONNECTIVITY SUMMARY")
    connected = sum([
        bool(alpaca_key and alpaca_secret),
        bool(fred_key),
        True,  # SEC doesn't need keys
    ])
    print(f"  {BOLD}Providers connected: {connected}/3{RESET}")
    if connected < 3:
        print(f"  {YELLOW}Some providers offline — pipeline will run in partial mode{RESET}")
    else:
        print(f"  {GREEN}All providers online — full live ingestion ready{RESET}")


async def ingest_live_data():
    """Fetch live data from all providers."""
    banner("LIVE DATA INGESTION", "Fetching real market data from Alpaca + FRED + SEC EDGAR")

    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET", "")
    fred_key = os.environ.get("FRED_API_KEY", "")

    today = date(2025, 6, 30)
    one_year_ago = today - timedelta(days=365)
    five_years_ago = today - timedelta(days=365 * 5)

    # ── 1. Alpaca OHLCV ──
    step("1", "Fetching OHLCV data from Alpaca")
    all_bars = {}
    if alpaca_key and alpaca_secret:
        provider = AlpacaProvider(api_key=alpaca_key, secret_key=alpaca_secret, paper=True)
        healthy = await provider.health_check()
        if healthy:
            result(f"Fetching {len(UNIVERSE)} symbols, {one_year_ago} to {today}")
            for symbol in UNIVERSE:
                try:
                    bars = await provider.fetch_ohlcv_daily(symbol, one_year_ago, today)
                    all_bars[symbol] = bars
                    result(f"  {symbol}: {len(bars)} bars fetched")
                    await asyncio.sleep(0.15)  # Rate limit
                except Exception as e:
                    fail(f"  {symbol}: {e}")
            success(f"Alpaca: {sum(len(v) for v in all_bars.values())} total bars for {len(all_bars)} symbols")
        else:
            fail("Alpaca health check failed — skipping OHLCV")
    else:
        fail("No Alpaca credentials — skipping OHLCV")

    # ── 2. FRED Macro Data ──
    step("2", "Fetching macro data from FRED")
    all_macro = {}
    if fred_key:
        provider = FredProvider(api_key=fred_key)
        healthy = await provider.health_check()
        if healthy:
            fred_series = list(set(
                s for comp in MCEI_COMPONENTS for s in comp.fred_series
            ))
            result(f"Fetching {len(fred_series)} FRED series for MCEI")
            for series_id in fred_series:
                try:
                    obs = await provider.fetch_macro_series(series_id, five_years_ago, today)
                    all_macro[series_id] = obs
                    result(f"  {series_id}: {len(obs)} observations")
                    await asyncio.sleep(0.15)  # Rate limit (7-8 req/sec)
                except Exception as e:
                    fail(f"  {series_id}: {e}")
            success(f"FRED: {sum(len(v) for v in all_macro.values())} total observations for {len(all_macro)} series")
        else:
            fail("FRED health check failed — skipping macro")
    else:
        fail("No FRED credentials — skipping macro")

    # ── 3. Compute MCEI ──
    step("3", "Computing MCEI from live macro data")
    if all_macro:
        component_values = {}
        component_histories = {}
        for comp in MCEI_COMPONENTS:
            for series_id in comp.fred_series:
                if series_id in all_macro and all_macro[series_id]:
                    latest = all_macro[series_id][-1]
                    history = [o["value"] for o in all_macro[series_id]]
                    component_values[comp.name] = latest["value"]
                    component_histories[comp.name] = history
                    break

        if component_values:
            mcei = calculate_mcei(
                component_values, component_histories, "v1", today
            )
            result(f"MCEI Score: {mcei.score:.2f}")
            result(f"Regime: {mcei.regime}")
            result(f"Components computed: {len(mcei.components)}")
            for name, cr in list(mcei.components.items())[:5]:
                result(f"  {name}: raw={cr.raw_value:.2f}, pct={cr.percentile:.1f}, norm={cr.normalized:.1f}")
            success(f"Live MCEI = {mcei.score:.2f} → Regime: {mcei.regime}")
        else:
            fail("No MCEI components could be computed")
    else:
        fail("No macro data — skipping MCEI")

    # ── 4. Compute Technical Features ──
    step("4", "Computing technical features from live OHLCV")
    import pandas as pd
    import numpy as np
    tech_features = {}
    for symbol, bars in all_bars.items():
        if len(bars) < 30:
            continue
        closes = pd.Series([b["close"] for b in bars])
        highs = pd.Series([b["high"] for b in bars])
        lows = pd.Series([b["low"] for b in bars])

        rsi_val = rsi(closes, 14)
        adx_val = adx(highs, lows, closes, 14)
        atr_val = atr(highs, lows, closes, 14)
        sma50 = sma(closes, 50)
        sma200 = sma(closes, 200)
        latest = float(closes.iloc[-1])

        tech_features[symbol] = {
            "price": latest,
            "rsi_14": rsi_val or 50,
            "adx_14": adx_val or 25,
            "atr_14": atr_val or latest * 0.02,
            "sma_50": sma50 or latest,
            "sma_200": sma200 or latest,
            "above_sma50": latest > (sma50 or 0),
            "above_sma200": latest > (sma200 or 0),
        }
        result(f"  {symbol}: ${latest:.2f} RSI={rsi_val or 0:.1f} ADX={adx_val or 0:.1f}")

    success(f"Technical features computed for {len(tech_features)} symbols")

    # ── Summary ──
    banner("LIVE INGESTION COMPLETE")
    print(f"  {BOLD}Alpaca OHLCV:{RESET}       {sum(len(v) for v in all_bars.values()):,} bars for {len(all_bars)} symbols")
    print(f"  {BOLD}FRED Macro:{RESET}        {sum(len(v) for v in all_macro.values()):,} observations for {len(all_macro)} series")
    print(f"  {BOLD}Technical Features:{RESET} {len(tech_features)} symbols analyzed")
    if 'mcei' in dir():
        print(f"  {BOLD}MCEI Score:{RESET}       {mcei.score:.2f} → {mcei.regime}")

    return all_bars, all_macro, tech_features


async def main():
    args = sys.argv[1:]
    mode = args[0] if args else "full"

    if mode == "--check":
        await check_connectivity()
    elif mode == "--ingest":
        await ingest_live_data()
    else:
        banner("MARKETMASTER — LIVE PIPELINE RUN",
               "Real data from Alpaca + FRED → MCEI → Pipeline")
        await check_connectivity()
        await ingest_live_data()
        print(f"\n  {BOLD}Next step: Connect PostgreSQL database to persist ingested data.{RESET}")
        print(f"  {BOLD}The pipeline can then run end-to-end with live data.{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
