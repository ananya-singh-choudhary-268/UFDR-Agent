"""
Centralized configuration management for UFDR Agent.

Uses Pydantic Settings for environment variable validation and type safety.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # =========================================================================
    # AI Model Configuration
    # =========================================================================
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key for AI model access"
    )
    gemini_model: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Gemini model identifier for LiteLLM"
    )
    
    # =========================================================================
    # Database Configuration
    # =========================================================================
    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL"
    )
    db_pool_min_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Minimum database connection pool size"
    )
    db_pool_max_size: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum database connection pool size"
    )
    db_command_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Database command timeout in seconds"
    )
    
    # =========================================================================
    # Redis Configuration
    # =========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # =========================================================================
    # Object Storage Configuration
    # =========================================================================
    s3_endpoint: str = Field(
        default="http://localhost:9000",
        description="S3-compatible storage endpoint (MinIO)"
    )
    s3_region: str = Field(
        default="us-east-1",
        description="S3 region"
    )
    s3_bucket: str = Field(
        default="ufdr-uploads",
        description="S3 bucket name for UFDR uploads"
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS/MinIO access key ID"
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS/MinIO secret access key"
    )
    
    # =========================================================================
    # Security Configuration
    # =========================================================================
    cors_allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )
    rate_limit_per_minute: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum requests per minute per IP address"
    )
    
    # =========================================================================
    # Logging Configuration
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # =========================================================================
    # Validation
    # =========================================================================
    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: str) -> str:
        """Ensure CORS origins are valid URLs."""
        if not v:
            return "http://localhost:3000"
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {valid_levels}")
        return upper_v
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


# =============================================================================
# Constants
# =============================================================================

class APIConstants:
    """API-related constants."""
    
    # Query limits
    MAX_QUERY_LENGTH = 10000
    MIN_QUERY_LENGTH = 1
    
    # Session
    SESSION_ID_PATTERN = r"^[a-zA-Z0-9\-_]{1,128}$"
    MAX_CHAT_HISTORY_LENGTH = 20
    SESSION_EXPIRATION_SECONDS = 3600
    
    # Rate limiting
    RATE_LIMIT_WINDOW_SECONDS = 60


class LogMessages:
    """Standardized log message templates."""
    
    # API
    REQUEST_RECEIVED = "Request received - Query: %s, Session: %s, Email: %s"
    RESPONSE_SENT = "Response sent - Session: %s, Status: %s"
    
    # Database
    DB_POOL_INIT_SUCCESS = "Database connection pool initialized successfully"
    DB_POOL_INIT_FAILED = "Failed to initialize database pool: %s"
    DB_POOL_CLOSED = "Database connection pool closed"
    
    # Redis
    REDIS_POOL_INIT_SUCCESS = "Redis connection pool initialized successfully"
    REDIS_POOL_INIT_FAILED = "Redis connection pool not initialized: %s"
    REDIS_POOL_CLOSED = "Redis connection pool closed"
    
    # Agent
    AGENT_CREATED = "Forensic agent created with %d tools"
    AGENT_QUERY = "Processing query: %s"
    AGENT_RESPONSE = "Agent response generated (%d characters)"


# =============================================================================
# Settings Singleton
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
