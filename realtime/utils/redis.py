"""
Redis connection pool management.

Provides async Redis connection pooling with proper
configuration, error handling, and graceful degradation.
"""

import logging
from typing import AsyncGenerator, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)

# Global connection pool
redis_pool: Optional[ConnectionPool] = None


class RedisManager:
    """
    Manages Redis connection pool.
    
    Provides connection pool management with graceful degradation
    when Redis is unavailable.
    """
    
    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
    
    @property
    def pool(self) -> Optional[ConnectionPool]:
        """Get the connection pool."""
        return self._pool
    
    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized and connected."""
        return self._pool is not None
    
    async def init_pool(self) -> Optional[ConnectionPool]:
        """
        Initialize the Redis connection pool.
        
        Returns:
            ConnectionPool if successful, None if Redis is unavailable
        """
        global redis_pool
        
        if self._pool is not None:
            logger.debug("Redis pool already initialized")
            return self._pool
        
        # Import settings here to avoid circular imports
        from config import get_settings
        settings = get_settings()
        
        redis_url = settings.redis_url
        logger.info("Initializing Redis connection pool: %s", redis_url)
        
        try:
            # Create connection pool with decode_responses for string handling
            pool = redis.ConnectionPool.from_url(redis_url, decode_responses=True)
            
            # Test connectivity
            async with redis.Redis(connection_pool=pool) as client:
                await client.ping()
            
            self._pool = pool
            redis_pool = pool  # Update global for backwards compatibility
            logger.info("Redis connection pool initialized successfully")
            return self._pool
            
        except Exception as e:
            # Redis is optional - log warning and continue without it
            self._pool = None
            redis_pool = None
            logger.warning("Redis connection pool not initialized: %s", e)
            return None
    
    async def close_pool(self) -> None:
        """Close the Redis connection pool."""
        global redis_pool
        
        if self._pool is not None:
            logger.info("Closing Redis connection pool")
            try:
                await self._pool.disconnect()
            except Exception as e:
                logger.warning("Error closing Redis pool: %s", e)
            finally:
                self._pool = None
                redis_pool = None


# Global Redis manager instance
_redis_manager = RedisManager()


# =============================================================================
# Public API (backwards compatible)
# =============================================================================

async def init_redis_pool() -> Optional[ConnectionPool]:
    """
    Initialize the Redis connection pool.
    
    This function is called during application startup.
    """
    return await _redis_manager.init_pool()


async def close_redis_pool() -> None:
    """
    Close the Redis connection pool.
    
    This function is called during application shutdown.
    """
    await _redis_manager.close_pool()


async def get_redis_client() -> AsyncGenerator[Optional[redis.Redis], None]:
    """
    FastAPI dependency to get a Redis client from the pool.
    
    Yields None if Redis is unavailable, allowing graceful degradation.
    
    Yields:
        redis.Redis client or None if unavailable
    """
    if _redis_manager.pool is None:
        # Redis unavailable - yield None for graceful degradation
        yield None
        return
    
    async with redis.Redis(connection_pool=_redis_manager.pool) as client:
        yield client
