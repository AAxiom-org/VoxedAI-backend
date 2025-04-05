"""
Models for chat-related database operations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class ChatMessageBase(BaseModel):
    """Base model for chat messages."""
    chat_session_id: str = Field(..., description="ID of the chat session this message belongs to")
    space_id: str = Field(..., description="ID of the notebook/space")
    user_id: str = Field(..., description="ID of the user who sent or received the message")
    content: str = Field(..., description="Content of the message")
    is_user: bool = Field(..., description="Whether the message is from the user (True) or AI (False)")
    workflow: Optional[List[Dict[str, Any]]] = Field(None, description="Workflow data for AI responses")
    reasoning: Optional[Dict[str, Any]] = Field(None, description="Reasoning data for AI responses")


class ChatMessageCreate(ChatMessageBase):
    """Model for creating a new chat message."""
    pass


class ChatMessage(ChatMessageBase):
    """Model for a chat message with ID and creation time."""
    id: str = Field(..., description="Unique identifier for the message")
    created_at: datetime = Field(..., description="When the message was created")

    class Config:
        orm_mode = True


class ChatSessionBase(BaseModel):
    """Base model for chat sessions."""
    user_id: str = Field(..., description="ID of the user who owns the session")
    space_id: str = Field(..., description="ID of the space/notebook")
    title: str = Field(..., description="Title of the chat session")


class ChatSessionCreate(ChatSessionBase):
    """Model for creating a new chat session."""
    pass


class ChatSession(ChatSessionBase):
    """Model for a chat session with ID and creation time."""
    id: str = Field(..., description="Unique identifier for the session")
    created_at: datetime = Field(..., description="When the session was created")

    class Config:
        orm_mode = True 