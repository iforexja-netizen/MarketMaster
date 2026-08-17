"""
Database session management with sync and async support.

Engines are lazily initialized so that importing this module doesn't
require a running database or installed drivers.
"""

from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from marketmaster.config.settings import settings

DATABASE_URL = settings.database_url


def _get_async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _get_sync_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


SYNC_DATABASE_URL = _get_sync_url(DATABASE_URL)
ASYNC_DATABASE_URL = _get_async_url(DATABASE_URL)

# Lazy singletons
_engine: Optional[Engine] = None
_async_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[sessionmaker] = None
_AsyncSessionLocal: Optional[async_sessionmaker] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(SYNC_DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_pre_ping=True)
    return _async_engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_async_session_factory() -> async_sessionmaker:
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _AsyncSessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for sync database sessions."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database sessions."""
    async with get_async_session_factory()() as session:
        yield session
