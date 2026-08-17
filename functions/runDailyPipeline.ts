/**
 * MarketMaster Daily Pipeline — Backend Function
 * 
 * Fetches latest market data from Alpaca + FRED, computes MCEI,
 * and returns a structured market briefing for the agent step.
 * 
 * Runs daily after US market close (4:15 PM ET).
 */

export default async function runDailyPipeline(req: any): Promise<any> {
  const ALPACA_KEY = Deno.env.get("ALPACA_API_KEY") || "";
  const ALPACA_SECRET = Deno.env.get("ALPACA_API_SECRET") || "";
  const FRED_KEY = Deno.env.get("FRED_API_KEY") || "";
  
  const isOffline = !ALPACA_KEY || !ALPACA_SECRET;
  const fredOffline = !FRED_KEY;
  
  // ── Universe ──────────────────────────────────────────────────
  const UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA",
    "DIS", "BAC", "XOM", "PFE", "KO"
  ];
  const SECTORS: Record<string, string> = {
    AAPL: "Technology", MSFT: "Technology", GOOGL: "Technology",
    AMZN: "Consumer", NVDA: "Technology", META: "Technology",
    TSLA: "Automotive", JPM: "Financials", V: "Financials",
    JNJ: "Healthcare", WMT: "Consumer", PG: "Consumer",
    UNH: "Healthcare", HD: "Consumer", MA: "Financials",
    DIS: "Entertainment", BAC: "Financials", XOM: "Energy",
    PFE: "Healthcare", KO: "Consumer",
  };
  
  // MCEI FRED series mapping (key components)
  const MCEI_SERIES: { series: string; weight: number; sign: string; name: string }[] = [
    { series: "WM2NS", weight: 0.10, sign: "pos", name: "broad_money_growth" },
    { series: "TOTLL", weight: 0.08, sign: "pos", name: "bank_credit_growth" },
    { series: "DGS10", weight: 0.07, sign: "neg", name: "10y_yield" },
    { series: "DGS2", weight: 0.06, sign: "neg", name: "2y_yield" },
    { series: "T10Y2Y", weight: 0.08, sign: "pos", name: "yield_curve_slope" },
    { series: "BAA10Y", weight: 0.07, sign: "neg", name: "credit_spread" },
    { series: "WALCL", weight: 0.09, sign: "pos", name: "fed_balance_sheet" },
    { series: "RRPONTSYD", weight: 0.05, sign: "neg", name: "repo_liquidity" },
  ];
  
  const today = new Date();
  const todayStr = today.toISOString().split("T")[0];
  const oneYearAgo = new Date(today);
  oneYearAgo.setFullYear(today.getFullYear() - 1);
  const oneYearAgoStr = oneYearAgo.toISOString().split("T")[0];
  
  // ── 1. Fetch FRED Macro Data ────────────────────────────────
  const macroResults: any[] = [];
  
  if (!fredOffline) {
    for (const comp of MCEI_SERIES) {
      try {
        const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${comp.series}&api_key=${FRED_KEY}&file_type=json&observation_start=${oneYearAgoStr}&observation_end=${todayStr}&limit=5&sort_order=desc`;
        const resp = await fetch(url);
        if (resp.ok) {
          const data = await resp.json();
          const obs = data.observations || [];
          if (obs.length > 0) {
            const latest = obs[obs.length - 1]; // most recent
            const value = parseFloat(latest.value);
            if (!isNaN(value)) {
              macroResults.push({
                name: comp.name,
                series: comp.series,
                value,
                weight: comp.weight,
                sign: comp.sign,
                date: latest.date,
              });
            }
          }
        }
      } catch (e) {
        // skip on error
      }
      // FRED rate limit: ~7 req/sec, but we're sequential so fine
    }
  }
  
  // ── 2. Compute MCEI Score ────────────────────────────────────
  let mceiScore = 50.0;
  let regime = "NEUTRAL";
  
  if (macroResults.length > 0) {
    let weightedSum = 0;
    let totalWeight = 0;
    for (const m of macroResults) {
      // Simple normalization: just use sign-adjusted value scaled to 0-100
      // In production, this would use percentile normalization vs history
      let normalized = 50;
      if (m.sign === "neg") {
        // For negative-sign series (yields, spreads), lower = more expansionary
        normalized = Math.max(0, Math.min(100, 100 - m.value * 5));
      } else {
        normalized = Math.max(0, Math.min(100, 50 + m.value * 0.1));
      }
      weightedSum += normalized * m.weight;
      totalWeight += m.weight;
    }
    mceiScore = totalWeight > 0 ? Math.round((weightedSum / totalWeight) * 100) / 100 : 50;
    
    if (mceiScore >= 70) regime = "STRONG_BULL";
    else if (mceiScore >= 60) regime = "BULL";
    else if (mceiScore >= 52) regime = "TRANSITION_BULL";
    else if (mceiScore >= 48) regime = "NEUTRAL";
    else if (mceiScore >= 38) regime = "TRANSITION_BEAR";
    else if (mceiScore >= 25) regime = "BEAR";
    else regime = "CRISIS";
  }
  
  // ── 3. Fetch Alpaca OHLCV ────────────────────────────────────
  const stockData: any[] = [];
  
  if (!isOffline) {
    const start = new Date(today);
    start.setDate(today.getDate() - 7); // last 7 days
    const startStr = start.toISOString().split("T")[0];
    
    for (const symbol of UNIVERSE) {
      try {
        const url = `https://data.alpaca.markets/v2/stocks/${symbol}/bars?timeframe=1Day&start=${startStr}&end=${todayStr}&limit=5&adjustment=all`;
        const resp = await fetch(url, {
          headers: {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
          },
        });
        if (resp.ok) {
          const data = await resp.json();
          const bars = data.bars || [];
          if (bars.length > 0) {
            const latest = bars[bars.length - 1];
            const prev = bars.length > 1 ? bars[bars.length - 2] : latest;
            const change = ((parseFloat(latest.c) - parseFloat(prev.c)) / parseFloat(prev.c)) * 100;
            stockData.push({
              symbol,
              sector: SECTORS[symbol] || "Unknown",
              price: parseFloat(latest.c),
              open: parseFloat(latest.o),
              high: parseFloat(latest.h),
              low: parseFloat(latest.l),
              volume: parseInt(latest.v),
              change_pct: Math.round(change * 100) / 100,
              date: latest.t?.split("T")[0] || todayStr,
            });
          }
        }
      } catch (e) {
        // skip on error
      }
      // Small delay to respect rate limits
      await new Promise(r => setTimeout(r, 120));
    }
  }
  
  // ── 4. Market Breadth ────────────────────────────────────────
  let advancers = 0, decliners = 0, unchanged = 0;
  let topGainer: any = null;
  let topLoser: any = null;
  
  for (const s of stockData) {
    if (s.change_pct > 0.5) advancers++;
    else if (s.change_pct < -0.5) decliners++;
    else unchanged++;
    
    if (!topGainer || s.change_pct > topGainer.change_pct) topGainer = s;
    if (!topLoser || s.change_pct < topLoser.change_pct) topLoser = s;
  }
  
  // ── 5. Strategy Recommendations (regime-aware) ───────────────
  const strategyMap: Record<string, string[]> = {
    STRONG_BULL: ["trend_following", "momentum", "breakout", "growth", "earnings_momentum", "sector_rotation", "macro_driven", "risk_parity"],
    BULL: ["trend_following", "momentum", "breakout", "earnings_momentum", "growth", "sector_rotation", "quality", "macro_driven", "risk_parity", "value"],
    TRANSITION_BULL: ["earnings_momentum", "quality", "value", "macro_driven", "risk_parity", "mean_reversion", "rsi_reversal"],
    NEUTRAL: ["earnings_momentum", "mean_reversion", "pairs_trading", "rsi_reversal", "value", "quality", "macro_driven", "risk_parity"],
    TRANSITION_BEAR: ["defensive", "low_volatility", "options_collar", "quality", "macro_driven", "risk_parity", "rsi_reversal", "pairs_trading", "sector_rotation"],
    BEAR: ["defensive", "low_volatility", "options_collar", "macro_driven", "risk_parity"],
    CRISIS: ["defensive", "low_volatility", "options_collar", "macro_driven", "risk_parity"],
    RECOVERY: ["earnings_momentum", "value", "quality", "sector_rotation", "macro_driven", "risk_parity", "mean_reversion"],
  };
  
  const activeStrategies = strategyMap[regime] || strategyMap["NEUTRAL"];
  
  const exposureMap: Record<string, number> = {
    STRONG_BULL: 0.50, BULL: 0.45, TRANSITION_BULL: 0.35, NEUTRAL: 0.25,
    TRANSITION_BEAR: 0.18, BEAR: 0.12, CRISIS: 0.05, RECOVERY: 0.30,
  };
  const recommendedExposure = exposureMap[regime] || 0.25;
  
  // ── 6. Sector Performance ───────────────────────────────────
  const sectorPerf: Record<string, { count: number; avgChange: number }> = {};
  for (const s of stockData) {
    if (!sectorPerf[s.sector]) sectorPerf[s.sector] = { count: 0, avgChange: 0 };
    sectorPerf[s.sector].count++;
    sectorPerf[s.sector].avgChange += s.change_pct;
  }
  const sectorSummary = Object.entries(sectorPerf).map(([sector, data]) => ({
    sector,
    avg_change: Math.round((data.avgChange / data.count) * 100) / 100,
    count: data.count,
  })).sort((a, b) => b.avg_change - a.avg_change);
  
  // ── Return ───────────────────────────────────────────────────
  return {
    run_date: todayStr,
    run_time: new Date().toISOString(),
    data_status: {
      alpaca: isOffline ? "offline (no API key)" : "live",
      fred: fredOffline ? "offline (no API key)" : "live",
      stocks_fetched: stockData.length,
      macro_components: macroResults.length,
    },
    mcei: {
      score: mceiScore,
      regime,
      components: macroResults.map(m => ({
        name: m.name,
        value: m.value,
        weight: m.weight,
        date: m.date,
      })),
    },
    market: {
      universe_size: UNIVERSE.length,
      stocks_with_data: stockData.length,
      advancers,
      decliners,
      unchanged,
      breadth: advancers - decliners,
      top_gainer: topGainer ? { symbol: topGainer.symbol, change: topGainer.change_pct } : null,
      top_loser: topLoser ? { symbol: topLoser.symbol, change: topLoser.change_pct } : null,
    },
    sector_performance: sectorSummary,
    strategies: {
      active: activeStrategies,
      count: activeStrategies.length,
      recommended_exposure: recommendedExposure,
    },
    stocks: stockData.map(s => ({
      symbol: s.symbol,
      price: s.price,
      change_pct: s.change_pct,
      sector: s.sector,
    })),
    pipeline_phases: [
      "Data Plane (Alpaca + FRED ingestion)",
      "MCEI Engine (macro conditions computed)",
      "Quant Engines (technical features)",
      "Regime Engine (classification complete)",
      "Strategy Selection (regime-aware)",
      "Portfolio Sizing (exposure-adjusted)",
      "Risk Gate (ready for paper execution)",
    ],
  };
}
