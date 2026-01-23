"""
Database operations for UFDR Agent.

Provides functions for saving and retrieving feedback data
from the PostgreSQL database.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .connection import get_db_connection

logger = logging.getLogger(__name__)


async def save_feedback(
    session_id: Optional[str],
    email_id: Optional[str],
    timestamp: str,
    query: str,
    generated_payload: Dict[str, Any],
    response: str
) -> bool:
    """
    Save feedback data to the feedback table in PostgreSQL.
    
    Args:
        session_id: Session identifier
        email_id: Email identifier
        timestamp: Timestamp in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        query: User's query
        generated_payload: The payload that was generated
        response: The AI response message
    
    Returns:
        True if successful, False otherwise
    """
    try:
        async with get_db_connection() as conn:
            # Convert payload to JSON string
            payload_json = json.dumps(generated_payload)
            
            # Parse ISO timestamp to datetime
            if isinstance(timestamp, str):
                timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                timestamp_dt = timestamp
            
            # Insert the feedback record
            await conn.execute(
                """
                INSERT INTO feedback (session_id, email_id, timestamp, query, generatedpayload, response)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                session_id,
                email_id,
                timestamp_dt,
                query,
                payload_json,
                response
            )
            
            logger.debug("Feedback saved for session: %s", session_id)
            return True
            
    except Exception as e:
        logger.error("Failed to save feedback: %s", e, exc_info=True)
        return False


async def get_feedback_by_session(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve feedback records by session_id.
    
    Args:
        session_id: Session identifier
    
    Returns:
        List of feedback records as dictionaries
    """
    try:
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, email_id, timestamp, query, generatedpayload, response
                FROM feedback
                WHERE session_id = $1
                ORDER BY timestamp DESC
                """,
                session_id
            )
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error("Failed to retrieve feedback by session: %s", e, exc_info=True)
        return []


async def get_feedback_by_email(email_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve feedback records by email_id.
    
    Args:
        email_id: Email identifier
    
    Returns:
        List of feedback records as dictionaries
    """
    try:
        async with get_db_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, email_id, timestamp, query, generatedpayload, response
                FROM feedback
                WHERE email_id = $1
                ORDER BY timestamp DESC
                """,
                email_id
            )
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error("Failed to retrieve feedback by email: %s", e, exc_info=True)
        return []
