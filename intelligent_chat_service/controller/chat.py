from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from schema import ChatRequest
from typing import Dict, Any
from core import OpenAIService, get_openai_service
from utils import logger
import asyncio
import json
from utils import get_orchestrator, standardize_event_type
import uuid
from datetime import datetime

chat_router = APIRouter(prefix="/ai", tags=["AI"])


@chat_router.post("/chat")
async def execute_workflow(
    request: ChatRequest, openai_service: OpenAIService = Depends(get_openai_service)
):
    try:
        return await handle_orchestration(request, openai_service)
    except Exception as e:
        logger.error(f"Error in execute_workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


async def handle_orchestration(
    request: ChatRequest, openai_service: OpenAIService
) -> StreamingResponse:
    """Handle the orchestration of the chat workflow

    Args:
        request (ChatRequest): The chat request containing user input and configuration
        openai_service (OpenAIService): Service for interacting with OpenAI

    Returns:
        StreamingResponse: A streaming response of events for the client

    Raises:
        HTTPException: If an error occurs during orchestration
    """
    try:
        # Get the appropriate orchestrator for this workflow
        try:
            orchestrator = get_orchestrator(request)
            if not orchestrator:
                logger.error(
                    f"Failed to get orchestrator for workflow: {request.workflow_name}"
                )
                raise HTTPException(
                    status_code=400, detail=f"Invalid workflow: {request.workflow_name}"
                )
        except ValueError as e:
            logger.error(f"Error creating orchestrator: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error creating orchestrator: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize workflow")

        orchestrator_execution_id = f"orchestration_{uuid.uuid4().hex[:8]}"
        message_counter = 0

        async def stream_orchestrator_results():
            queue = asyncio.Queue()

            async def orchestrator_callback(event_type: str, data: Dict[str, Any]):
                nonlocal message_counter
                timestamp = datetime.now().isoformat() + "Z"

                standard_event_type = standardize_event_type(event_type)

                metadata = {
                    **data,
                    "orchestration_id": orchestrator_execution_id,
                    "timestamp": timestamp,
                }

                if event_type == "content_chunk":
                    agent_info = data.get("agent", "system")
                    chunk = data.get("chunk", "")

                    event_message = {
                        "event": "ui:content:chunk",
                        "data": {
                            "chunk": chunk,
                            "agent": agent_info,
                            "message_id": f"message_{message_counter}",
                            "orchestration_id": orchestrator_execution_id,
                            "timestamp": timestamp,
                        },
                    }

                    await queue.put(f"data: {json.dumps(event_message)}\n\n")

                elif event_type == "step_update" or "role" in data:
                    message_counter += 1
                    metadata["agent_id"] = data.get("step")
                    metadata["message_id"] = f"message_{message_counter}"

                    event_data = {"event": standard_event_type, "data": metadata}
                    await queue.put(f"data: {json.dumps(event_data)}\n\n")
                else:
                    event_data = {"event": standard_event_type, "data": metadata}
                    await queue.put(f"data: {json.dumps(event_data)}\n\n")

            orchestrator_task = asyncio.create_task(
                orchestrator.execute(
                    context={
                        "input_text": request.user_input,
                        "data_sources": request.selected_sources,
                        "thread_id": request.config.thread_id,
                    },
                    openai_service=openai_service,
                    callback=orchestrator_callback,
                )
            )

            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(queue.get(), timeout=1)
                        yield chunk
                        queue.task_done()
                    except asyncio.TimeoutError:
                        if orchestrator_task.done():
                            if queue.empty():
                                break
                        continue
            finally:
                if not orchestrator_task.done():
                    orchestrator_task.cancel()
                    try:
                        await orchestrator_task
                    except asyncio.CancelledError:
                        pass

                final_event = {
                    "event": "ui:orchestrator:complete",
                    "data": {
                        "name": orchestrator.name,
                        "orchestration_id": orchestrator_execution_id,
                        "timestamp": datetime.now().isoformat() + "Z",
                    },
                }
                yield f"data: {json.dumps(final_event)}\n\n"

        return StreamingResponse(
            content=stream_orchestrator_results(), media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in handle_orchestration: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
