"""
Analytics API routes for forensic data analysis.

Provides endpoints for processing forensic queries using AI-powered analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis
from fastapi import APIRouter, Depends, Request

from config import get_settings
from schemas.objects import AnalyticsPayload, AnalyticsResponse
from utils.ai.agent import create_forensic_agent
from utils.chat_session import get_chat_history, save_chat_message
from utils.db import save_feedback
from utils.redis import get_redis_client
from utils.time import is_valid_timestamp

# Get settings and configure logging
settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


@router.options("/analytics")
async def analytics_options():
    """Handle CORS preflight OPTIONS requests."""
    return {"status": "ok"}


@router.post("/analytics", response_model=AnalyticsResponse)
async def analytics_endpoint(
    request: Request,
    payload: AnalyticsPayload,
    redis_client: Optional[redis.Redis] = Depends(get_redis_client),
) -> AnalyticsResponse:
    """
    Process forensic analysis queries.
    
    Receives a query from the frontend, processes it through the AI agent,
    and returns a detailed forensic analysis response.
    
    Args:
        request: FastAPI request object (for request ID)
        payload: AnalyticsPayload containing query and metadata
        redis_client: Redis client for chat history (optional)
    
    Returns:
        AnalyticsResponse with AI-generated forensic analysis
    """
    # Get request ID for tracing
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        # Log incoming request (avoid logging full query for privacy)
        logger.info(
            "[%s] Analytics request - Session: %s, Email: %s, Query length: %d",
            request_id,
            payload.session_id,
            payload.email_id,
            len(payload.query)
        )
        
        # Prepare timestamp
        now = datetime.now(timezone.utc)
        current_timestamp = payload.current_timestamp or now.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Validate timestamp format if provided
        if payload.current_timestamp and not is_valid_timestamp(current_timestamp):
            logger.warning("[%s] Invalid timestamp format: %s", request_id, current_timestamp)
            return AnalyticsResponse(
                status="error",
                message=f"Invalid timestamp format. Expected: YYYY-MM-DDTHH:MM:SSZ. Example: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                response={},
                session_id=payload.session_id,
                status_code=400,
                request_id=request_id
            )
        
        # Create AI agent
        agent = await create_forensic_agent()
        
        # Get chat history from Redis for context (graceful degradation)
        chat_history = await get_chat_history(redis_client, payload.session_id)
        
        # Process the query using the agent
        logger.debug("[%s] Processing query with agent", request_id)
        agent_response = await agent.analyze_forensic_data(payload.query, chat_history)
        
        # Log response summary
        logger.info(
            "[%s] Agent response generated - Length: %d chars",
            request_id,
            len(agent_response)
        )
        
        # Save chat message to Redis (best-effort, non-blocking failure)
        try:
            await save_chat_message(
                redis_client, payload.session_id, payload.query, agent_response
            )
        except Exception as chat_error:
            logger.warning(
                "[%s] Failed to save chat message: %s",
                request_id,
                chat_error
            )
        
        # Build successful response
        response_data = AnalyticsResponse(
            message=agent_response,
            status="success",
            response={"query": payload.query},
            session_id=payload.session_id,
            status_code=200,
            request_id=request_id
        )
        
        # Save feedback to database (best-effort, non-blocking failure)
        try:
            await save_feedback(
                session_id=payload.session_id,
                email_id=payload.email_id,
                timestamp=current_timestamp,
                query=payload.query,
                generated_payload={"query": payload.query},
                response=agent_response
            )
            logger.debug("[%s] Feedback saved to database", request_id)
        except Exception as db_error:
            logger.warning(
                "[%s] Failed to save feedback to database: %s",
                request_id,
                db_error
            )
        
        return response_data
        
    except ValueError as e:
        # Validation errors
        logger.warning("[%s] Validation error: %s", request_id, e)
        return AnalyticsResponse(
            status="error",
            message=f"Validation error: {str(e)}",
            response={},
            session_id=payload.session_id,
            status_code=400,
            request_id=request_id
        )
        
    except ConnectionError as e:
        # Database/Redis connection errors
        logger.error("[%s] Connection error: %s", request_id, e)
        return AnalyticsResponse(
            status="error",
            message="Service temporarily unavailable. Please try again.",
            response={},
            session_id=payload.session_id,
            status_code=503,
            request_id=request_id
        )
        
    except Exception as e:
        # Unexpected errors - log full traceback
        logger.exception("[%s] Unexpected error processing analytics request", request_id)
        return AnalyticsResponse(
            status="error",
            message="An error occurred while processing your request. Please try again.",
            response={},
            session_id=payload.session_id,
            status_code=500,
            request_id=request_id
        )
