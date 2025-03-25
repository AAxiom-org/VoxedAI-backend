"""
Schema definitions for agent operations.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class AgentRequest(BaseModel):
    """Schema for agent request."""

class AgentResponse(BaseModel):
    """Schema for agent response."""
    success: bool = Field(..., 
                         description="Whether the agent execution was successful")
                         
