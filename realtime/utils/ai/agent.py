"""
Forensic AI Agent implementation.

Provides the ForensicAgent class that uses LiteLLM with Gemini
for AI-powered forensic data analysis.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

from config import get_settings, APIConstants
from tools.apps import app_tool
from tools.browsing_history import browsing_history_tool
from tools.call_logs import call_log_tool
from tools.contacts import contact_tool
from tools.location import location_tool
from tools.messages import message_tool
from utils.prompts.Forensic_agent import forensic_agent_instructions

logger = logging.getLogger(__name__)


class ForensicAgent:
    """
    AI-powered forensic data analysis agent.
    
    Uses Gemini via LiteLLM for deep forensic analysis of UFDR report data.
    
    Attributes:
        agent: The underlying Agent instance with forensic tools
    """
    
    def __init__(self):
        """
        Initialize the Forensic Agent with Gemini model via LiteLLM.
        
        Raises:
            ValueError: If GEMINI_API_KEY is not configured
        """
        settings = get_settings()
        
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Define available tools
        tools = [
            location_tool,
            app_tool,
            call_log_tool,
            message_tool,
            browsing_history_tool,
            contact_tool
        ]
        
        # Log tool registration
        logger.info("Initializing ForensicAgent with %d tools", len(tools))
        for tool in tools:
            logger.debug("  - %s: %s", tool.name, tool.description[:50])
        
        # Create the agent
        self.agent = Agent(
            name="ForensicAnalyst",
            instructions=forensic_agent_instructions,
            model=LitellmModel(
                model=settings.gemini_model,
                api_key=settings.gemini_api_key
            ),
            tools=tools,
        )
        
        logger.info("ForensicAgent initialized successfully")
    
    async def analyze_forensic_data(
        self,
        user_query: str,
        chat_history: str = "",
        data_chunks: Optional[List[str]] = None
    ) -> str:
        """
        Process a forensic query by analyzing provided UFDR report data.
        
        Args:
            user_query: The investigator's question or analysis request
            chat_history: Prior conversation context for follow-up queries
            data_chunks: Optional list of UFDR report data chunks to analyze
        
        Returns:
            The agent's forensic analysis response as a string
        
        Raises:
            ValueError: If query exceeds maximum length
        """
        # Validate query length
        if len(user_query) > APIConstants.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query exceeds maximum length of {APIConstants.MAX_QUERY_LENGTH} characters"
            )
        
        if data_chunks is None:
            data_chunks = []
        
        # Build structured prompt with context
        sections: List[str] = []
        
        if chat_history:
            sections.append(
                "Prior Chat Context (use this for follow-ups, pronouns, and continuity):\n"
                + chat_history
            )
        
        sections.append("Current User Query:\n" + user_query)
        sections.append(
            f"Forensic Data Summary: {len(data_chunks)} data chunks available for analysis."
        )
        
        query_with_context = "\n\n".join(sections)
        
        # Log query processing (avoid logging full query content)
        logger.debug(
            "Processing query - Length: %d, History: %d chars, Chunks: %d",
            len(user_query),
            len(chat_history),
            len(data_chunks)
        )
        
        # Run the agent
        result = await Runner.run(self.agent, query_with_context)
        
        logger.debug("Agent response generated - Length: %d chars", len(result.final_output))
        
        return result.final_output


# =============================================================================
# Factory Function
# =============================================================================

async def create_forensic_agent() -> ForensicAgent:
    """
    Factory function to create a ForensicAgent instance.
    
    Returns:
        Configured ForensicAgent instance
    """
    return ForensicAgent()
