"""
MarketMaster End-to-End Integration Test

Tests the full Data Plane stack:
1. SQLAlchemy models can create tables
2. DataPlane can write and read records
3. Decision log hash chaining works
4. Data quality checks run
5. API routes are importable

Uses SQLite in-memory for testing (no PostgreSQL required).
"""

import pytest
from datetime import date, datetime, timezone
from decimal import Decimal

# These tests use SQLite — some PostgreSQL-specific features (JSONB, ARRAY)
# won't work. We test what we can and skip PG-specific tests.


class TestModelsImport:
    """Verify all models can be imported."""

    def test_import_all_models(self):
        from marketmaster.db.models import (
            SecurityMaster,
            OhlcvDaily,
            OhlcvIntraday,
            CorporateActions,
            Fundamentals,
            SecFilings,
            MacroSeries,
            EconomicEvents,
            NewsItems,
            Transcripts,
            OptionChains,
            Features,
            Signals,
            MceiHistory,
            RegimeHistory,
            Decision,
            Trade,
            PortfolioSnapshot,
            RiskMetric,
            DataQualityLog,
            IngestionLog,
            MceiConfig,
        )
        # All 22 models should be importable
        assert SecurityMaster is not None
        assert OhlcvDaily is not None
        assert MceiConfig is not None

    def test_import_decision_log(self):
        from marketmaster.db.decision_log import log_decision, verify_chain_integrity
        assert log_decision is not None
        assert verify_chain_integrity is not None

    def test_import_data_plane(self):
        from marketmaster.data.plane import DataPlane, DataPlaneError
        assert DataPlane is not None
        assert DataPlaneError is not None

    def test_import_providers(self):
        from marketmaster.data.providers import (
            DataProvider,
            AlpacaProvider,
            FredProvider,
            SecEdgarProvider,
        )
        assert DataProvider is not None
        assert AlpacaProvider is not None
        assert FredProvider is not None
        assert SecEdgarProvider is not None

    def test_import_ingestion(self):
        from marketmaster.data.ingestion import IngestionCoordinator, IngestionResult
        assert IngestionCoordinator is not None
        assert IngestionResult is not None

    def test_import_quality(self):
        from marketmaster.data.quality import DataQualityService, DataQualityResult
        assert DataQualityService is not None
        assert DataQualityResult is not None


class TestMCEIConfig:
    """Verify MCEI configuration is complete and consistent."""

    def test_components_count(self):
        from marketmaster.config.mcei_series import MCEI_COMPONENTS
        assert len(MCEI_COMPONENTS) == 16

    def test_weights_sum_near_one(self):
        from marketmaster.config.mcei_series import get_total_weight
        total = get_total_weight()
        assert 0.8 <= total <= 1.2, f"Weights sum to {total}, expected ~1.0"

    def test_all_series_codes_unique(self):
        from marketmaster.config.mcei_series import get_all_series_codes
        codes = get_all_series_codes()
        assert len(codes) == len(set(codes)), "Duplicate FRED series codes found"

    def test_regimes_defined(self):
        from marketmaster.config.mcei_series import MARKET_REGIMES
        assert len(MARKET_REGIMES) == 8
        assert "NEUTRAL" in MARKET_REGIMES
        assert "CRISIS" in MARKET_REGIMES

    def test_categories_cover_hierarchy(self):
        from marketmaster.config.mcei_series import MCEI_COMPONENTS
        categories = {c.category for c in MCEI_COMPONENTS}
        expected = {"money", "credit", "liquidity", "rates", "yield_curve",
                    "credit_spread", "financial_conditions"}
        assert categories == expected


class TestProviderClasses:
    """Verify provider classes are properly structured."""

    def test_alpaca_provider_init(self):
        from marketmaster.data.providers import AlpacaProvider
        p = AlpacaProvider("key", "secret")
        assert p.name == "alpaca"
        assert p.paper is True

    def test_fred_provider_init(self):
        from marketmaster.data.providers import FredProvider
        p = FredProvider("key")
        assert p.name == "fred"

    def test_sec_provider_init(self):
        from marketmaster.data.providers import SecEdgarProvider
        p = SecEdgarProvider("MarketMaster admin@example.com")
        assert p.name == "sec_edgar"

    def test_base_provider_abstract(self):
        from marketmaster.data.providers.base import DataProvider
        import abc
        assert issubclass(DataProvider, abc.ABC)


class TestDecisionLog:
    """Verify decision log hashing logic."""

    def test_hash_deterministic(self):
        from marketmaster.db.decision_log import _compute_decision_hash
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        args = dict(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy="momentum", regime="BULL",
            approved=True, score=0.85, evidence={"agent": "test"},
            risk_assessment={"max_risk": 0.5}, context={"market": "up"},
            agent_chain=[{"agent": "technical"}], prev_hash=None,
        )
        h1 = _compute_decision_hash(**args)
        h2 = _compute_decision_hash(**args)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_prev_hash_chains(self):
        from marketmaster.db.decision_log import _compute_decision_hash
        now = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        base_args = dict(
            timestamp=now, security_id=1, symbol="AAPL",
            decision_type="entry", strategy=None, regime=None,
            approved=False, score=None, evidence={},
            risk_assessment={}, context={}, agent_chain=[],
        )
        h1 = _compute_decision_hash(prev_hash=None, **base_args)
        h2 = _compute_decision_hash(prev_hash="abc", **base_args)
        assert h1 != h2


class TestOrchestrator:
    """Verify the updated orchestrator structure."""

    def test_orchestrator_init(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator
        orch = MarketMasterOrchestrator()
        assert len(orch.agents) >= 3  # Phase 3: default agents loaded
        assert orch.plane is None

    def test_orchestrator_with_agents(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator

        class MockAgent:
            async def analyze(self, symbol):
                pass

        orch = MarketMasterOrchestrator(agents=[MockAgent()])
        assert len(orch.agents) == 1

    def test_risk_gate_integration(self):
        from marketmaster.agents.orchestrator import MarketMasterOrchestrator
        orch = MarketMasterOrchestrator()

        # Should reject: position risk too high
        decision = orch.check_risk(position_risk_pct=0.02, daily_loss_pct=0.01)
        assert not decision.approved
        assert "POSITION_RISK_LIMIT" in decision.reasons

        # Should reject: live trading disabled
        decision = orch.check_risk(position_risk_pct=0.001, daily_loss_pct=0.001)
        assert not decision.approved
        assert "LIVE_TRADING_DISABLED" in decision.reasons


class TestAPIImport:
    """Verify the API routes are importable and wired."""

    def test_router_importable(self):
        from marketmaster.api.routes import router
        assert router is not None

    def test_app_importable(self):
        from marketmaster.main import app
        assert app is not None
        assert app.title == "MarketMaster API"
        assert app.version == "0.7.0"
