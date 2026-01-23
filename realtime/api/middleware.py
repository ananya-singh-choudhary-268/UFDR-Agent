"""
Middleware components for the UFDR API.

Provides request logging, timing, and request ID generation.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request context including:
    - Unique request ID for tracing
    - Request timing
    - Request/response logging
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with context."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Store request ID in state for access in route handlers
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.perf_counter()
        
        # Log incoming request (minimal info to avoid logging sensitive data)
        logger.info(
            "[%s] %s %s",
            request_id,
            request.method,
            request.url.path
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            # Log exception with request ID for tracing
            logger.exception("[%s] Unhandled exception: %s", request_id, exc)
            raise
        
        # Calculate request duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Add headers for tracing
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log response
        logger.info(
            "[%s] %s %s -> %d (%.2fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )
        
        return response


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    
    logger.info("Logging configured at level: %s", log_level)
