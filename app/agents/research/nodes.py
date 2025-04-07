"""
Node classes for research digests.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class DigestNode(BaseModel):
    """Node representing a research digest."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    all_links: List[str] = Field(default_factory=list)
    related_note_ids: List[str] = Field(default_factory=list)
    space_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
