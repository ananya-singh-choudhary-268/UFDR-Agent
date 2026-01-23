"""
Time validation utilities.

Provides functions for validating and parsing timestamp formats.
"""

from datetime import datetime
from typing import Optional


def is_valid_timestamp(date_string: Optional[str]) -> bool:
    """
    Validate that a string is a valid ISO 8601 timestamp.
    
    Expected format: YYYY-MM-DDTHH:MM:SSZ
    Example: 2025-01-23T10:30:00Z
    
    Args:
        date_string: The timestamp string to validate
    
    Returns:
        True if the timestamp is valid, False otherwise
    """
    if not date_string:
        return False
    
    try:
        datetime.strptime(str(date_string), "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def parse_timestamp(date_string: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp string to a datetime object.
    
    Supports formats:
    - YYYY-MM-DDTHH:MM:SSZ (UTC)
    - YYYY-MM-DDTHH:MM:SS+HH:MM (with timezone)
    
    Args:
        date_string: The timestamp string to parse
    
    Returns:
        datetime object if parsing succeeds, None otherwise
    """
    if not date_string:
        return None
    
    # Try standard format first
    try:
        return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass
    
    # Try ISO format with timezone
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        return None


def get_current_timestamp() -> str:
    """
    Get the current UTC timestamp in ISO 8601 format.
    
    Returns:
        Current timestamp as string (YYYY-MM-DDTHH:MM:SSZ)
    """
    from datetime import timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")