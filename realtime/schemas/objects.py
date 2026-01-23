"""
Pydantic models for API request/response validation.

Provides input validation, sanitization, and type safety for API endpoints.
"""

import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Request Validation Constants
# =============================================================================

MAX_QUERY_LENGTH = 10000
MIN_QUERY_LENGTH = 1
MAX_SESSION_ID_LENGTH = 128
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]+$")
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


# =============================================================================
# Request Models
# =============================================================================

class AnalyticsPayload(BaseModel):
    """
    Request payload for the analytics endpoint.
    
    Attributes:
        query: The forensic analysis query from the user
        current_timestamp: Optional ISO timestamp for time context
        session_id: Optional session identifier for conversation tracking
        email_id: Optional email identifier for user tracking
    """
    
    query: str = Field(
        ...,
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description="Forensic analysis query"
    )
    current_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SSZ)"
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=MAX_SESSION_ID_LENGTH,
        description="Session identifier for conversation tracking"
    )
    email_id: Optional[str] = Field(
        default=None,
        max_length=320,  # Max email length per RFC 5321
        description="User email identifier"
    )
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize the query string."""
        # Strip leading/trailing whitespace
        v = v.strip()
        
        if len(v) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} character(s)")
        
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query must not exceed {MAX_QUERY_LENGTH} characters")
        
        return v
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate session ID format."""
        if v is None:
            return v
        
        v = v.strip()
        
        if not v:
            return None
        
        if len(v) > MAX_SESSION_ID_LENGTH:
            raise ValueError(f"Session ID must not exceed {MAX_SESSION_ID_LENGTH} characters")
        
        if not SESSION_ID_PATTERN.match(v):
            raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
        
        return v
    
    @field_validator("email_id")
    @classmethod
    def validate_email_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v is None:
            return v
        
        v = v.strip().lower()
        
        if not v:
            return None
        
        if not EMAIL_PATTERN.match(v):
            raise ValueError("Invalid email format")
        
        return v
    
    @field_validator("current_timestamp")
    @classmethod
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        """Validate timestamp format."""
        if v is None:
            return v
        
        v = v.strip()
        
        if not v:
            return None
        
        if not TIMESTAMP_PATTERN.match(v):
            raise ValueError("Timestamp must be in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ")
        
        return v


# =============================================================================
# Response Models
# =============================================================================

class AnalyticsResponse(BaseModel):
    """
    Response model for the analytics endpoint.
    
    Attributes:
        message: The AI-generated forensic analysis response
        status: Status of the request (success/error)
        response: Additional response metadata
        session_id: Session identifier for conversation tracking
        status_code: HTTP status code
        request_id: Optional request ID for tracing
    """
    
    message: str = Field(
        ...,
        description="AI-generated forensic analysis response"
    )
    status: str = Field(
        ...,
        description="Response status (success/error)"
    )
    response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier"
    )
    status_code: Optional[int] = Field(
        default=200,
        ge=100,
        le=599,
        description="HTTP status code"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing"
    )


class UFDRUploadResponse(BaseModel):
    """
    Response model for UFDR file upload endpoint.
    
    Attributes:
        status: Status of the upload (success/error)
        file_info: Information about the uploaded file
        file_id: Unique identifier for the uploaded file
        status_code: HTTP status code
    """
    
    status: str = Field(
        ...,
        description="Upload status (success/error)"
    )
    file_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the uploaded file"
    )
    file_id: Optional[str] = Field(
        default=None,
        description="Unique file identifier"
    )
    status_code: Optional[int] = Field(
        default=200,
        ge=100,
        le=599,
        description="HTTP status code"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoints."""
    
    status: str = Field(..., description="Health status")
    service: Optional[str] = Field(default=None, description="Service name")
    checks: Optional[Dict[str, bool]] = Field(default=None, description="Individual health checks")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")