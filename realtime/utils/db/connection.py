"""
Database connection pool management.

Provides async database connection pooling using asyncpg with proper
configuration, error handling, and logging.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database connection pool.
    
    Provides thread-safe connection pool management with proper
    initialization, health checking, and cleanup.
    """
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the connection pool."""
        return self._pool
    
    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._pool is not None
    
    async def init_pool(self) -> asyncpg.Pool:
        """
        Initialize the database connection pool.
        
        Uses settings from config module for connection parameters.
        
        Returns:
            asyncpg.Pool: Initialized connection pool
            
        Raises:
            ValueError: If DATABASE_URL is not configured
            ConnectionError: If database connection fails
        """
        if self._pool is not None:
            logger.debug("Database pool already initialized")
            return self._pool
        
        # Import settings here to avoid circular imports
        from config import get_settings
        settings = get_settings()
        
        database_url = settings.database_url
        if not database_url:
            raise ValueError("DATABASE_URL is not configured")
        
        try:
            self._pool = await asyncpg.create_pool(
                database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                command_timeout=settings.db_command_timeout
            )
            logger.info(
                "Database connection pool initialized (min=%d, max=%d)",
                settings.db_pool_min_size,
                settings.db_pool_max_size
            )
            return self._pool
            
        except Exception as e:
            logger.error("Failed to initialize database pool: %s", e)
            raise ConnectionError(f"Database connection failed: {e}") from e
    
    async def close_pool(self) -> None:
        """Close the database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
    
    async def get_pool(self) -> asyncpg.Pool:
        """
        Get the database connection pool, initializing if necessary.
        
        Returns:
            asyncpg.Pool: Active connection pool
            
        Raises:
            ConnectionError: If pool cannot be initialized
        """
        if self._pool is None:
            await self.init_pool()
        return self._pool
    
    @asynccontextmanager
    async def connection(self):
        """
        Context manager to get a database connection from the pool.
        
        Yields:
            asyncpg.Connection: Database connection
            
        Example:
            async with db_manager.connection() as conn:
                await conn.execute("SELECT 1")
        """
        pool = await self.get_pool()
        async with pool.acquire() as connection:
            yield connection


# Global database manager instance
_db_manager = DatabaseManager()


# =============================================================================
# Public API (backwards compatible)
# =============================================================================

async def init_db_pool() -> asyncpg.Pool:
    """Initialize the database connection pool."""
    return await _db_manager.init_pool()


async def close_db_pool() -> None:
    """Close the database connection pool."""
    await _db_manager.close_pool()


async def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool."""
    return await _db_manager.get_pool()


@asynccontextmanager
async def get_db_connection():
    """Context manager to get a database connection from the pool."""
    async with _db_manager.connection() as conn:
        yield conn
