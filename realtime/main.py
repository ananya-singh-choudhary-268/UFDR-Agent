"""
UFDR Real-time API

FastAPI application entry point with proper configuration,
security, and health check endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.analytics import router as analytics_router
from api.uploads.routes import router as uploads_router
from api.middleware import RequestContextMiddleware, setup_logging
from config import get_settings
from utils.db import init_db_pool, close_db_pool
from utils.redis import init_redis_pool, close_redis_pool

# Get settings
settings = get_settings()

# Configure logging first
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for application startup and shutdown.
    
    Manages database and Redis connection pools.
    """
    # Startup
    logger.info("Starting UFDR Real-time API...")
    
    try:
        await init_db_pool()
        logger.info("Database pool initialized")
    except Exception as e:
        logger.error("Failed to initialize database pool: %s", e)
        # Continue without database - some endpoints may still work
    
    try:
        await init_redis_pool()
        logger.info("Redis pool initialized")
    except Exception as e:
        logger.error("Failed to initialize Redis pool: %s", e)
        # Continue without Redis - chat history won't be available
    
    logger.info("UFDR Real-time API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down UFDR Real-time API...")
    
    await close_redis_pool()
    await close_db_pool()
    
    logger.info("UFDR Real-time API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="UFDR Real-time API",
    description="Real-time analytics API for UFDR Agent - Digital Forensic Analysis",
    version="1.1.0",
    lifespan=lifespan
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# Add request context middleware (logging, timing, request ID)
app.add_middleware(RequestContextMiddleware)

# Add CORS middleware with environment-based configuration
cors_origins = settings.cors_origins_list
logger.info("Configuring CORS for origins: %s", cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    
    Logs the error with request context and returns a safe error response.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("[%s] Unhandled exception: %s", request_id, exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred",
            "request_id": request_id
        }
    )


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - basic API information."""
    return {
        "message": "UFDR Real-time API is running",
        "version": "1.1.0"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns basic health status without checking dependencies.
    Use /ready for full readiness check.
    """
    return {
        "status": "healthy",
        "service": "ufdr-api"
    }


@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes and orchestrators.
    
    Checks that all dependencies are available and the service
    is ready to accept traffic.
    """
    from utils.db import get_db_pool
    from utils.redis import redis_pool
    
    checks = {
        "database": False,
        "redis": False
    }
    
    # Check database
    try:
        pool = await get_db_pool()
        if pool:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["database"] = True
    except Exception as e:
        logger.warning("Database readiness check failed: %s", e)
    
    # Check Redis
    try:
        if redis_pool:
            import redis.asyncio as aioredis
            async with aioredis.Redis(connection_pool=redis_pool) as client:
                await client.ping()
            checks["redis"] = True
    except Exception as e:
        logger.warning("Redis readiness check failed: %s", e)
    
    # Determine overall status
    all_ready = all(checks.values())
    
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "status": "ready" if all_ready else "degraded",
            "checks": checks
        }
    )


# =============================================================================
# API Router Registration
# =============================================================================

# Include the analytics router
app.include_router(analytics_router, prefix="/api")

# Include the uploads router
app.include_router(uploads_router)


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )