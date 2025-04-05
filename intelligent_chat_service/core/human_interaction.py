import asyncio
from typing import Dict, Any, Optional, Callable
import uuid
from datetime import datetime
import json
from utils import logger


class HumanInteractionService:
    """Service to handle human-in-the-loop interactions."""

    def __init__(self):
        # Store pending interactions: {interaction_id: event}
        self.pending_interactions = {}
        # Store received human responses: {interaction_id: response}
        self.human_responses = {}
        # Event to notify when a human response is received
        self.response_events = {}
        # Debug counter to track interaction lifecycle
        self._debug_counter = 0

    async def request_human_input(
        self, agent_id: str, question: str, context: Dict[str, Any], timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Request input from a human user.

        Args:
            agent_id: The ID of the agent making the request
            question: The question to ask the human
            context: Additional context about the request
            timeout: Timeout in seconds to wait for a response

        Returns:
            The human's response or None if timed out
        """
        # Use the provided interaction_id from the event if available
        if (
            context
            and isinstance(context, dict)
            and context.get("event_data")
            and context["event_data"].get("interaction_id")
        ):
            interaction_id = context["event_data"]["interaction_id"]
            logger.info(f"Using interaction_id from event: {interaction_id}")
        else:
            # Generate a clean interaction ID without any prefix to avoid confusion
            interaction_id = str(uuid.uuid4())
            self._debug_counter += 1
            logger.info(
                f"Generated new interaction_id: {interaction_id} (#{self._debug_counter})"
            )

        # Create an event that will be set when the human responds
        self.response_events[interaction_id] = asyncio.Event()

        # Store the interaction details with the interaction_id explicitly included
        self.pending_interactions[interaction_id] = {
            "agent_id": agent_id,
            "question": question,
            "context": context,
            "timestamp": datetime.now().isoformat() + "Z",
            "interaction_id": interaction_id,
            "counter": self._debug_counter,
        }

        logger.info(
            f"Agent {agent_id} requested human input: {question} (interaction_id: {interaction_id}, counter: {self._debug_counter})"
        )

        # Debug: Log all pending interactions
        logger.info(
            f"Current pending interactions: {list(self.pending_interactions.keys())}"
        )

        try:
            # Wait for the human to respond or timeout
            await asyncio.wait_for(self.response_events[interaction_id].wait(), timeout)

            # Get the response
            response = self.human_responses.get(interaction_id)

            logger.info(f"Response received for interaction {interaction_id}")

            return {
                "response": response,
                "interaction_id": interaction_id,
                "status": "success",
            }

        except asyncio.TimeoutError:
            logger.warning(
                f"Human input request timed out for interaction {interaction_id}"
            )
            # Keep in pending interactions for possible later handling
            return {
                "response": None,
                "interaction_id": interaction_id,
                "status": "timeout",
            }

        finally:
            # Don't clean up the pending interaction here - we'll let the submit_human_response method handle that
            # Just clean up the event
            if interaction_id in self.response_events:
                del self.response_events[interaction_id]

    def submit_human_response(self, interaction_id: str, response: Any) -> bool:
        """
        Submit a response from a human.

        Args:
            interaction_id: The ID of the interaction
            response: The human's response

        Returns:
            True if the interaction exists and the response was accepted, False otherwise
        """
        # Debug: Log all pending interactions at time of submission
        logger.info(f"Submitting response for interaction_id: {interaction_id}")
        logger.info(
            f"Available pending interactions: {list(self.pending_interactions.keys())}"
        )

        # Check if there's any possibly matching interaction
        partial_matches = [
            i for i in self.pending_interactions.keys() if interaction_id in i
        ]
        if partial_matches and interaction_id not in self.pending_interactions:
            logger.warning(
                f"Possible partial matches for ID: {interaction_id}: {partial_matches}"
            )

            # Use the partial match if there's exactly one
            if len(partial_matches) == 1:
                actual_id = partial_matches[0]
                logger.info(
                    f"Using partial match {actual_id} for requested ID {interaction_id}"
                )
                interaction_id = actual_id

        if interaction_id not in self.pending_interactions:
            logger.warning(
                f"Received response for unknown interaction: {interaction_id}"
            )
            return False

        # Store the response
        self.human_responses[interaction_id] = response

        # Notify that a response has been received
        if interaction_id in self.response_events:
            self.response_events[interaction_id].set()
            logger.info(f"Event set for interaction {interaction_id}")
        else:
            logger.warning(f"No event found for interaction {interaction_id}")

        # Now remove from pending interactions after response is processed
        counter = self.pending_interactions[interaction_id].get("counter", 0)
        del self.pending_interactions[interaction_id]
        logger.info(
            f"Human response received and interaction {interaction_id} (#{counter}) removed from pending"
        )

        return True

    def get_pending_interactions(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending interactions."""
        return self.pending_interactions

    def cleanup_responses(self) -> int:
        """
        Clean up stored responses that are no longer needed.
        Returns the number of responses cleaned up.
        """
        # Find response ids that don't have a corresponding pending interaction
        to_remove = []
        for response_id in self.human_responses.keys():
            if response_id not in self.pending_interactions:
                to_remove.append(response_id)

        # Remove the responses
        for response_id in to_remove:
            del self.human_responses[response_id]

        logger.info(f"Cleaned up {len(to_remove)} human responses")
        return len(to_remove)


# Singleton instance
_human_interaction_service = None


def get_human_interaction_service() -> HumanInteractionService:
    """Get or create a human interaction service instance."""
    global _human_interaction_service
    if _human_interaction_service is None:
        _human_interaction_service = HumanInteractionService()
    return _human_interaction_service
