"""
Schema definitions for agent operations.
"""
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator


class QueryResult(BaseModel):
    """Schema for a single query result/source."""
    id: str = Field(..., description="Unique identifier for the source")
    content: str = Field(..., description="Content of the source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the source")


class ThinkingStep(BaseModel):
    """Schema for a thinking/reasoning step."""
    step: int = Field(..., description="Step number in the reasoning process")
    thinking: str = Field(..., description="The reasoning/thinking content")


class AgentRequest(BaseModel):
    """Schema for agent request."""
    space_id: str = Field(..., description="ID of the space the query pertains to")
    query: str = Field(..., description="The user's question or command")
    view: Optional[str] = Field(None, description="The content the user is actively working with")
    stream: bool = Field(False, description="Whether to stream the response")


class AgentResponse(BaseModel):
    """Schema for agent response."""
    success: bool = Field(..., description="Whether the agent execution was successful")
    response: str = Field(..., description="The response text (answer)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional information about the response")
    
    @property
    def sources(self) -> List[QueryResult]:
        """Get the sources used for this response."""
        return self.metadata.get("sources", [])
    
    @property
    def thinking(self) -> List[ThinkingStep]:
        """Get the thinking steps for this response."""
        return self.metadata.get("thinking", [])
    
    @property
    def query_time_ms(self) -> int:
        """Get the query execution time in milliseconds."""
        return self.metadata.get("query_time_ms", 0)
                         
