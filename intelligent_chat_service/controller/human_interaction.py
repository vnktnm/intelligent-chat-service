from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from core import get_human_interaction_service, HumanInteractionService
from utils import logger

human_router = APIRouter(prefix="/ai/human", tags=["Human Interaction"])


class HumanResponse(BaseModel):
    """Schema for human response to agent questions."""

    interaction_id: str = Field(
        ..., description="ID of the interaction being responded to"
    )
    response: Any = Field(..., description="Human's response to the agent's question")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the response"
    )


class InteractionQuery(BaseModel):
    """Schema for querying interactions."""

    agent_id: Optional[str] = None


@human_router.post("/response")
async def submit_human_response(
    response: HumanResponse,
    human_service: HumanInteractionService = Depends(get_human_interaction_service),
):
    """Submit a human response to an agent's question."""
    logger.info(f"Received human response for interaction {response.interaction_id}")

    # Debug: Log the pending interactions
    pending_interactions = human_service.get_pending_interactions()
    logger.info(f"Current pending interactions: {list(pending_interactions.keys())}")

    # Attempt to find a matching interaction ID
    interaction_id = response.interaction_id

    # Try to find an exact or partial match
    if interaction_id not in pending_interactions:
        partial_matches = [
            i for i in pending_interactions.keys() if interaction_id in i
        ]
        if partial_matches:
            logger.info(
                f"Found partial matches for {interaction_id}: {partial_matches}"
            )
            if len(partial_matches) == 1:
                matched_id = partial_matches[0]
                logger.info(f"Using closest match: {matched_id}")
                interaction_id = matched_id

    success = human_service.submit_human_response(
        interaction_id=interaction_id, response=response.response
    )

    if not success:
        logger.warning(
            f"Failed to submit response for interaction {interaction_id} - not found or completed"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Interaction {interaction_id} not found or already completed",
        )

    return {
        "status": "success",
        "message": "Response submitted successfully",
        "interaction_id": interaction_id,
    }


@human_router.get("/pending")
async def get_pending_interactions(
    human_service: HumanInteractionService = Depends(get_human_interaction_service),
):
    """Get all pending interactions waiting for human input."""
    pending = human_service.get_pending_interactions()
    return {"interactions": pending, "count": len(pending)}


@human_router.post("/cleanup")
async def cleanup_interactions(
    human_service: HumanInteractionService = Depends(get_human_interaction_service),
):
    """Manually trigger cleanup of completed interactions."""
    cleaned_count = human_service.cleanup_responses()
    return {"status": "success", "cleaned_count": cleaned_count}
