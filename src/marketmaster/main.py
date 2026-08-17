"""
MarketMaster FastAPI Application

Entry point for the MarketMaster API.

Phase 1: Data Plane — schema, providers, ingestion, quality
Phase 5: Risk + Paper Trading — pipeline, technical/fundamental factors, features

All data access flows through the DataPlane coordinator.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from marketmaster.api.routes import router as phase1_router
from marketmaster.api.phase2_routes import phase2_router
from marketmaster.api.phase3_routes import phase3_router
from marketmaster.api.phase4_routes import phase4_router
from marketmaster.api.phase5_routes import phase5_router
from marketmaster.api.phase6_routes import phase6_router
from marketmaster.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    print(f"[MarketMaster] Starting in {settings.app_env} mode")
    print(f"[MarketMaster] Live trading: {'ENABLED' if settings.enable_live_trading else 'DISABLED'}")
    print(f"[MarketMaster] Phase: MCEI + Quant Engines (Phase 2)")

    if settings.app_env == "development":
        from marketmaster.db.base import Base
        from marketmaster.db.session import get_engine

        Base.metadata.create_all(bind=get_engine())
        print("[MarketMaster] Database tables created (development mode)")

    yield

    # Shutdown
    print("[MarketMaster] Shutting down")


app = FastAPI(
    title="MarketMaster API",
    version="0.7.0",
    description="AI-native market intelligence, quantitative research, portfolio/risk, and trading platform",
    lifespan=lifespan,
)

# Include Phase 1 routes (data plane)
app.include_router(phase1_router, prefix="/api/v1")

# Include Phase 2 routes (MCEI + quant engines)
app.include_router(phase2_router, prefix="/api/v1")
app.include_router(phase3_router, prefix="/api/v1")
app.include_router(phase4_router, prefix="/api/v1")
app.include_router(phase5_router, prefix="/api/v1")
app.include_router(phase6_router, prefix="/api/v1")


@app.get("/health")
def health():
    """Root health check."""
    return {
        "status": "ok",
        "service": "marketmaster",
        "version": "0.7.0",
        "phase": "learning_system",
        "live_trading": settings.enable_live_trading,
    }
