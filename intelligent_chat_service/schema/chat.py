from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import config


class Config(BaseModel):
    thread_id: str = Field(..., description="Thread ID")
    client_id: str = Field(..., description="Client ID")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    human_in_the_loop: bool = Field(
        config.HITL_ENABLED, description="Enable human-in-the-loop"
    )


class ChatRequest(BaseModel):
    workflow_name: str = Field(..., description="Workflow name")
    user_input: str = Field(..., description="User input")
    model: str = Field(config.OPENAI_DEFAULT_MODEL, description="Model name")
    temperature: Optional[float] = Field(
        config.OPENAI_DEFAULT_TEMPERATURE, description="Temperature"
    )
    stream: bool = Field(True, description="Stream response")
    config: Optional[Config] = Field(None, description="Configuration")
    selected_sources: List[str] = Field([], description="Selected sources")
