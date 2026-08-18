import { createClientFromRequest } from "npm:@base44/sdk@0.8.31";

Deno.serve(async (req: Request) => {
  try {
    const base44 = createClientFromRequest(req);
    const ALPACA_KEY = Deno.env.get("ALPACA_API_KEY") || "";
    const ALPACA_SECRET = Deno.env.get("ALPACA_API_SECRET") || "";
    const FRED_KEY = Deno.env.get("FRED_API_KEY") || "";
    
    const isOffline = !ALPACA_KEY || !ALPACA_SECRET;
    const fredOffline = !FRED_KEY;
    
    const UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "DIS", "BAC", "XOM", "PFE", "KO"];
    const SECTORS: Record<string, string> = {
      AAPL: "Technology", MSFT: "Technology", GOOGL: "Technology", AMZN: "Consumer", NVDA: "Technology", META: "Technology", TSLA: "Automotive", JPM: "Financials", V: "Financials", JNJ: "Healthcare", WMT: "Consumer", PG: "Consumer", UNH: "Healthcare", HD: "Consumer", MA: "Financials", DIS: "Entertainment", BAC: "Financials", XOM: "Energy", PFE: "Healthcare", KO: "Consumer",
    };
    
    const MCEI_SERIES = [
      { series: "WM2NS", weight: 0.10, sign: "pos", name: "broad_money_growth" },
      { series: "DGS10", weight: 0.07, sign: "neg", name: "10y_yield" },
      { series: "T10Y2Y", weight: 0.08, sign: "pos", name: "yield_curve_slope" },
      { series: "BAA10Y", weight: 0.07, sign: "neg", name: "credit_spread" },
    ];
    
    const today = new Date();
    const todayStr = today.toISOString().split("T")[0];
    const oneYearAgo = new Date(today);
    oneYearAgo.setFullYear(today.getFullYear() - 1);
    const oneYearAgoStr = oneYearAgo.toISOString().split("T")[0];
    
    // 1. FRED Macro
    const macroResults: any[] = [];
    if (!fredOffline) {
      for (const comp of MCEI_SERIES) {
        try {
          const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${comp.series}&api_key=${FRED_KEY}&file_type=json&observation_start=${oneYearAgoStr}&observation_end=${todayStr}&limit=5&sort_order=desc`;
          const ctrl = new AbortController();
          const timeout = setTimeout(() => ctrl.abort(), 8000);
          const resp = await fetch(url, { signal: ctrl.signal });
          clearTimeout(timeout);
          if (resp.ok) {
            const data = await resp.json();
            const obs = data.observations || [];
            if (obs.length > 0) {
              const latest = obs[obs.length - 1];
              const value = parseFloat(latest.value);
              if (!isNaN(value)) {
                macroResults.push({ name: comp.name, series: comp.series, value, weight: comp.weight, sign: comp.sign, date: latest.date });
              }
            }
          }
        } catch (e) { /* skip */ }
      }
    }
    
    // 2. MCEI
    let mceiScore = 50.0;
    let regime = "NEUTRAL";
    if (macroResults.length > 0) {
      let ws = 0, tw = 0;
      for (const m of macroResults) {
        let n = m.sign === "neg" ? Math.max(0, Math.min(100, 100 - m.value * 5)) : Math.max(0, Math.min(100, 50 + m.value * 0.1));
        ws += n * m.weight; tw += m.weight;
      }
      mceiScore = tw > 0 ? Math.round((ws / tw) * 100) / 100 : 50;
      if (mceiScore >= 70) regime = "STRONG_BULL";
      else if (mceiScore >= 60) regime = "BULL";
      else if (mceiScore >= 52) regime = "TRANSITION_BULL";
      else if (mceiScore >= 48) regime = "NEUTRAL";
      else if (mceiScore >= 38) regime = "TRANSITION_BEAR";
      else if (mceiScore >= 25) regime = "BEAR";
      else regime = "CRISIS";
    }
    
    // 3. Alpaca — fetch symbols using IEX feed (free tier, no SIP subscription required)
    const stockData: any[] = [];
    if (!isOffline) {
      const start = new Date(today); start.setDate(today.getDate() - 7);
      const startStr = start.toISOString().split("T")[0];
      for (const symbol of UNIVERSE) {
        try {
          const url = `https://data.alpaca.markets/v2/stocks/${symbol}/bars?timeframe=1Day&start=${startStr}&end=${todayStr}&limit=5&adjustment=all&feed=iex`;
          const ctrl = new AbortController();
          const timeout = setTimeout(() => ctrl.abort(), 8000);
          const resp = await fetch(url, {
            headers: { "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET },
            signal: ctrl.signal
          });
          clearTimeout(timeout);
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
                change_pct: Math.round(change * 100) / 100,
                date: (latest.t || todayStr).split("T")[0]
              });
            }
          }
        } catch (e) { /* skip */ }
        await new Promise(r => setTimeout(r, 100));
      }
    }
    
    // 4. Market breadth
    let advancers = 0, decliners = 0;
    let topGainer: any = null, topLoser: any = null;
    for (const s of stockData) {
      if (s.change_pct > 0.5) advancers++;
      else if (s.change_pct < -0.5) decliners++;
      if (!topGainer || s.change_pct > topGainer.change_pct) topGainer = s;
      if (!topLoser || s.change_pct < topLoser.change_pct) topLoser = s;
    }
    
    // 5. Strategy selection
    const strategyMap: Record<string, string[]> = {
      STRONG_BULL: ["trend_following", "momentum", "breakout", "growth", "earnings_momentum", "sector_rotation", "macro_driven", "risk_parity"],
      BULL: ["trend_following", "momentum", "breakout", "earnings_momentum", "growth", "sector_rotation", "quality", "macro_driven", "risk_parity", "value"],
      TRANSITION_BULL: ["earnings_momentum", "quality", "value", "macro_driven", "risk_parity", "mean_reversion", "rsi_reversal"],
      NEUTRAL: ["earnings_momentum", "mean_reversion", "pairs_trading", "rsi_reversal", "value", "quality", "macro_driven", "risk_parity"],
      TRANSITION_BEAR: ["defensive", "low_volatility", "options_collar", "quality", "macro_driven", "risk_parity", "rsi_reversal", "pairs_trading"],
      BEAR: ["defensive", "low_volatility", "options_collar", "macro_driven", "risk_parity"],
      CRISIS: ["defensive", "low_volatility", "options_collar", "macro_driven", "risk_parity"],
      RECOVERY: ["earnings_momentum", "value", "quality", "sector_rotation", "macro_driven", "risk_parity", "mean_reversion"],
    };
    const activeStrategies = strategyMap[regime] || strategyMap["NEUTRAL"];
    const exposureMap: Record<string, number> = { STRONG_BULL: 0.50, BULL: 0.45, TRANSITION_BULL: 0.35, NEUTRAL: 0.25, TRANSITION_BEAR: 0.18, BEAR: 0.12, CRISIS: 0.05, RECOVERY: 0.30 };
    const recommendedExposure = exposureMap[regime] || 0.25;
    
    return Response.json({
      run_date: todayStr,
      data_status: {
        alpaca: isOffline ? "offline" : "live",
        fred: fredOffline ? "offline" : "live",
        stocks_fetched: stockData.length,
        macro_components: macroResults.length,
        feed: "iex"
      },
      mcei: {
        score: mceiScore,
        regime,
        components: macroResults.map(m => ({ name: m.name, value: m.value, date: m.date }))
      },
      market: {
        stocks_with_data: stockData.length,
        advancers,
        decliners,
        breadth: advancers - decliners,
        top_gainer: topGainer ? { symbol: topGainer.symbol, change: topGainer.change_pct } : null,
        top_loser: topLoser ? { symbol: topLoser.symbol, change: topLoser.change_pct } : null
      },
      strategies: {
        active: activeStrategies,
        count: activeStrategies.length,
        recommended_exposure: recommendedExposure
      },
      stocks: stockData.map(s => ({ symbol: s.symbol, price: s.price, change_pct: s.change_pct, sector: s.sector })),
    });
  } catch (e: any) {
    return Response.json({ error: e.message }, { status: 500 });
  }
});
