-- ============================================================================
-- MarketMaster — Seed Equity Universe
-- ============================================================================
-- Initial universe for testing. Not comprehensive — the full universe
-- (S&P 500 + Russell 1000 + major ETFs) will be loaded via Alpaca API.
-- ============================================================================

-- Major ETFs (proxy for indices + sectors)
INSERT INTO security_master (symbol, name, asset_class, exchange, sector, industry) VALUES
('SPY',  'SPDR S&P 500 ETF Trust',                  'etf', 'ARCA', 'Broad Market', 'Large Blend'),
('QQQ',  'Invesco QQQ Trust',                       'etf', 'NASDAQ', 'Broad Market', 'Large Growth'),
('IWM',  'iShares Russell 2000 ETF',                'etf', 'ARCA', 'Broad Market', 'Small Blend'),
('DIA',  'SPDR Dow Jones Industrial Average ETF',   'etf', 'ARCA', 'Broad Market', 'Large Value'),
('VTI',  'Vanguard Total Stock Market ETF',         'etf', 'ARCA', 'Broad Market', 'Large Blend'),
('XLK',  'Technology Select Sector SPDR',           'etf', 'ARCA', 'Technology', 'Technology'),
('XLF',  'Financial Select Sector SPDR',             'etf', 'ARCA', 'Financials', 'Financials'),
('XLE',  'Energy Select Sector SPDR',                'etf', 'ARCA', 'Energy', 'Energy'),
('XLV',  'Health Care Select Sector SPDR',           'etf', 'ARCA', 'Healthcare', 'Healthcare'),
('XLY',  'Consumer Discretionary Select Sector SPDR','etf', 'ARCA', 'Consumer Discretionary', 'Consumer Discretionary'),
('XLP',  'Consumer Staples Select Sector SPDR',     'etf', 'ARCA', 'Consumer Staples', 'Consumer Staples'),
('XLI',  'Industrial Select Sector SPDR',           'etf', 'ARCA', 'Industrials', 'Industrials'),
('XLB',  'Materials Select Sector SPDR',            'etf', 'ARCA', 'Materials', 'Materials'),
('XLU',  'Utilities Select Sector SPDR',            'etf', 'ARCA', 'Utilities', 'Utilities'),
('XLRE', 'Real Estate Select Sector SPDR',          'etf', 'ARCA', 'Real Estate', 'Real Estate'),
('XLC',  'Communication Services Select Sector SPDR','etf', 'ARCA', 'Communication Services', 'Communication Services'),

-- Bond / Rate ETFs
('TLT',  'iShares 20+ Year Treasury Bond ETF',      'etf', 'ARCA', 'Fixed Income', 'Long Government'),
('IEF',  'iShares 7-10 Year Treasury Bond ETF',     'etf', 'ARCA', 'Fixed Income', 'Intermediate Government'),
('SHY',  'iShares 1-3 Year Treasury Bond ETF',      'etf', 'ARCA', 'Fixed Income', 'Short Government'),
('HYG',  'iShares iBoxx $ High Yield Corp Bond ETF','etf', 'ARCA', 'Fixed Income', 'High Yield'),
('LQD',  'iShares iBoxx $ Inv Grade Corp Bond ETF', 'etf', 'ARCA', 'Fixed Income', 'Corporate'),

-- Commodity / Alternative
('GLD',  'SPDR Gold Shares',                        'etf', 'ARCA', 'Commodities', 'Gold'),
('SLV',  'iShares Silver Trust',                   'etf', 'ARCA', 'Commodities', 'Silver'),
('USO',  'United States Oil Fund',                  'etf', 'ARCA', 'Commodities', 'Oil'),
('UNG',  'United States Natural Gas Fund',          'etf', 'ARCA', 'Commodities', 'Natural Gas'),

-- Volatility
('UVXY', 'ProShares Ultra VIX Short-Term Futures',  'etf', 'ARCA', 'Volatility', 'Volatility'),
('VIX',  'CBOE Volatility Index',                   'index', 'CBOE', 'Volatility', 'Volatility'),

-- Major mega-cap equities
('AAPL', 'Apple Inc.',                              'equity', 'NASDAQ', 'Technology', 'Consumer Electronics'),
('MSFT', 'Microsoft Corporation',                   'equity', 'NASDAQ', 'Technology', 'Software—Infrastructure'),
('GOOGL','Alphabet Inc. Class A',                    'equity', 'NASDAQ', 'Communication Services', 'Internet Content & Information'),
('AMZN', 'Amazon.com Inc.',                         'equity', 'NASDAQ', 'Consumer Discretionary', 'Internet Retail'),
('NVDA', 'NVIDIA Corporation',                      'equity', 'NASDAQ', 'Technology', 'Semiconductors'),
('META', 'Meta Platforms Inc.',                    'equity', 'NASDAQ', 'Communication Services', 'Internet Content & Information'),
('TSLA', 'Tesla Inc.',                             'equity', 'NASDAQ', 'Consumer Discretionary', 'Auto Manufacturers'),
('JPM',  'JPMorgan Chase & Co.',                   'equity', 'NYSE', 'Financials', 'Banks—Diversified'),
('V',    'Visa Inc.',                              'equity', 'NYSE', 'Financials', 'Credit Services'),
('BRK.B','Berkshire Hathaway Inc. Class B',         'equity', 'NYSE', 'Financials', 'Insurance—Diversified'),
('UNH',  'UnitedHealth Group Inc.',                'equity', 'NYSE', 'Healthcare', 'Healthcare Plans'),
('XOM',  'Exxon Mobil Corporation',                'equity', 'NYSE', 'Energy', 'Oil & Gas Integrated'),
('JNJ',  'Johnson & Johnson',                      'equity', 'NYSE', 'Healthcare', 'Drug Manufacturers'),
('WMT',  'Walmart Inc.',                           'equity', 'NYSE', 'Consumer Staples', 'Discount Stores'),
('LLY',  'Eli Lilly and Company',                  'equity', 'NYSE', 'Healthcare', 'Drug Manufacturers'),
('AVGO', 'Broadcom Inc.',                          'equity', 'NASDAQ', 'Technology', 'Semiconductors'),
('PG',   'Procter & Gamble Company',               'equity', 'NYSE', 'Consumer Staples', 'Household & Personal Products'),
('MA',   'Mastercard Incorporated',                'equity', 'NYSE', 'Financials', 'Credit Services'),
('HD',   'Home Depot Inc.',                        'equity', 'NYSE', 'Consumer Discretionary', 'Home Improvement Retail'),
('COST', 'Costco Wholesale Corporation',           'equity', 'NASDAQ', 'Consumer Staples', 'Discount Stores')

ON CONFLICT (symbol, exchange) DO NOTHING;
