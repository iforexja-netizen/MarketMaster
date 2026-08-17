"""
MarketMaster application settings.

Loads from environment variables and/or .env file.
All values have sensible defaults for development mode.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_env: str = "development"

    # Database
    database_url: str = "postgresql+asyncpg://marketmaster:marketmaster@localhost:5432/marketmaster"
    database_url_sync: str = "postgresql+psycopg://marketmaster:marketmaster@localhost:5432/marketmaster"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Alpaca
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_paper: bool = True
    alpaca_data_url: str = "https://data.alpaca.markets/v2"
    alpaca_paper_trading_url: str = "https://paper-api.alpaca.markets"

    # FRED
    fred_api_key: str = ""

    # SEC EDGAR
    sec_user_agent: str = "MarketMaster iforexja@gmail.com"

    # AI Provider
    ai_provider: str = ""
    ai_api_key: str = ""
    ai_model: str = "gpt-4o"

    # Trading Controls — LIVE TRADING DISABLED BY DEFAULT
    enable_live_trading: bool = False

    # Risk Parameters
    max_position_risk_pct: float = 0.005
    max_daily_loss_pct: float = 0.02
    max_portfolio_risk_pct: float = 0.10
    max_sector_exposure_pct: float = 0.30
    max_single_position_pct: float = 0.10

    # Data Ingestion
    ingestion_batch_size: int = 1000
    ingestion_rate_limit_rps: int = 10

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
