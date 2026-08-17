-- ============================================================================
-- MarketMaster — MCEI Component → FRED Series Mapping
-- ============================================================================
-- This maps each MCEI component to its FRED series code(s).
-- The ingestion layer uses this to pull the right data.
-- Signs are aligned so HIGHER = more expansionary.
-- ============================================================================

-- This is a reference/config table, not raw data. Stored as a meta table.
CREATE TABLE IF NOT EXISTS mcei_config (
    id              SERIAL PRIMARY KEY,
    component_name  VARCHAR(64)   NOT NULL UNIQUE,
    display_name    VARCHAR(128),
    fred_series     VARCHAR(64)[] NOT NULL,     -- one or more FRED series codes
    sign            VARCHAR(4)    NOT NULL DEFAULT 'pos',  -- pos = higher is expansionary, neg = higher is contractionary
    transform       VARCHAR(32)   DEFAULT 'pct_yoy',      -- pct_yoy, pct_qoq, level, zscore, percentile
    weight          DECIMAL(6,4)  DEFAULT 0.0,            -- weight in MCEI composite
    description     TEXT,
    category        VARCHAR(32),                          -- money, credit, liquidity, rates, yield_curve, credit_spread, financial_conditions
    is_active       BOOLEAN       DEFAULT TRUE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- MCEI Component Definitions
-- ============================================================================
-- Money & Credit
INSERT INTO mcei_config (component_name, display_name, fred_series, sign, transform, weight, description, category) VALUES
('broad_money_growth', 'Broad Money Growth (M2 YoY)', '{"WM2NS"}', 'pos', 'pct_yoy', 0.10, 'M2 money supply year-over-year growth rate. Higher = more expansionary monetary conditions.', 'money'),

('bank_credit_growth', 'Bank Credit Growth (YoY)', '{"TOTBKCR"}', 'pos', 'pct_yoy', 0.08, 'Total bank credit, all commercial banks. Year-over-year growth. Expansionary when growing.', 'credit'),

('ci_lending', 'Commercial & Industrial Loan Growth', '{"BUSLOANS"}', 'pos', 'pct_yoy', 0.07, 'C&I loans at all commercial banks. YoY growth reflects business investment appetite.', 'credit'),

('consumer_credit', 'Consumer Credit Growth', "{\"TOTCI\"}", 'pos', 'pct_yoy', 0.05, 'Total consumer credit outstanding. Growth signals consumer spending capacity.', 'credit'),

-- Liquidity
('fed_balance_sheet', 'Fed Balance Sheet (YoY)', '{"WALCL"}', 'pos', 'pct_yoy', 0.08, 'Federal Reserve total assets. Expansionary when growing (QE), contractionary when shrinking (QT).', 'liquidity'),

('treasury_liquidity', 'Treasury General Account / TGA', '{"WTREGEN"}', 'neg', 'pct_yoy', 0.04, 'Treasury General Account balance. Drawdowns inject liquidity into the system.', 'liquidity'),

('rrp_usage', 'Reverse Repo Facility Usage', '{"RRPONTSYD"}', 'neg', 'level', 0.03, 'ON RRP facility usage. High usage drains liquidity; declining usage is expansionary.', 'liquidity'),

-- Rates
('fed_funds_rate', 'Federal Funds Rate', '{"DFF"}', 'neg', 'level', 0.07, 'Effective federal funds rate. Lower rates = more expansionary.', 'rates'),

('real_rates', '10-Year Real Yield (TIPS)', '{"DGS10", "DFII10"}', 'neg', 'spread', 0.06, '10-year nominal minus 10-year breakeven inflation. Lower real yields = more expansionary.', 'rates'),

-- Yield Curve
('yield_curve_slope', 'Yield Curve Slope (10Y-2Y)', '{"DGS10", "DGS2"}', 'pos', 'spread', 0.08, '10-year minus 2-year Treasury yield spread. Steeper curve = expansionary expectations.', 'yield_curve'),

('yield_curve_3m10y', 'Yield Curve (10Y-3M)', '{"DGS10", "DGS3MO"}', 'pos', 'spread', 0.05, '10-year minus 3-month Treasury yield spread. Classic recession predictor when inverted.', 'yield_curve'),

-- Credit Spreads
('credit_spread_ig', 'IG Credit Spread (BAA-AAA)', '{"BAA", "AAA"}', 'neg', 'spread', 0.06, 'Moody''s BAA minus AAA corporate yield spread. Widening spreads = tighter conditions.', 'credit_spread'),

('credit_spread_hy', 'High Yield OAS', '{"BAMLH0A0HYM2"}', 'neg', 'level', 0.05, 'ICE BofA US High Yield Index option-adjusted spread. Tighter spreads = expansionary.', 'credit_spread'),

-- Financial Conditions
('financial_conditions', 'Chicago Fed National Financial Conditions Index', '{"NFCI"}', 'neg', 'level', 0.06, 'Chicago Fed NFCI. Negative = accommodative, positive = restrictive.', 'financial_conditions'),

('financial_conditions_leveraged', 'NFCI Leverage Subindex', '{"NFCILEVERAGE"}', 'neg', 'level', 0.03, 'Leverage component of NFCI. Captures risk appetite in funding markets.', 'financial_conditions'),

('dxy', 'US Dollar Index', '{"DTWEXBGS"}', 'neg', 'pct_yoy', 0.03, 'Trade-weighted dollar index. Weaker dollar = more expansionary global liquidity.', 'financial_conditions');

-- ============================================================================
-- Verify total weights sum approximately to 1.0
-- (Some rounding is expected; weights are configurable and will be
--  optimized via walk-forward testing)
-- ============================================================================
-- Expected: 0.10+0.08+0.07+0.05+0.08+0.04+0.03+0.07+0.06+0.08+0.05+0.06+0.05+0.06+0.03+0.03 = 0.94
-- Remaining ~0.06 reserved for future components or can be redistributed.

COMMENT ON TABLE mcei_config IS 'MCEI component configuration: maps each macro component to FRED series, sign alignment, transform, and weight. Weights are initial estimates — must be validated with walk-forward testing.';
