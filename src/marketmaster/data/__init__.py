"""
MarketMaster Data Package

The Data Plane is the single source of truth for all market data.
Every agent, engine, and service reads from here — no independent data fetching.

Usage:
    from marketmaster.data.plane import DataPlane

    plane = DataPlane(db_session)
    prices = plane.get_ohlcv_daily(security_id=1, start_date="2024-01-01")
    macro = plane.get_macro_series("WALCL", realtime_date="2024-06-30")
"""
