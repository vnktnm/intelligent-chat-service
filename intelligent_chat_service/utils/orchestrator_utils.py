from orchestrator import Orchestrator, IDiscoveryOrchestrator
from typing import Optional
from schema import ChatRequest


def get_orchestrator(request: ChatRequest) -> Orchestrator:
    orchestrators = {"idiscovery_orchestrator": IDiscoveryOrchestrator}

    if request.workflow_name not in orchestrators:
        raise ValueError(
            f"Workflow {request.workflow_name} not found. Available workflows: {list(orchestrators.keys())}"
        )

    return orchestrators[request.workflow_name](request)


def standardize_event_type(event_type: str) -> str:
    """Convert internal event types to UI friendly event types with proper namespacing"""
    event_mapping = {
        # Existing mappings
        "orchestrator_start": "ui:orchestrator:start",
        "orchestrator_progress": "ui:orchestrator:progress",
        "orchestrator_complete": "ui:orchestrator:complete",
        "step_update": "ui:step:update",
        "step_complete": "ui:step:complete",
        "collaboration_turn": "ui:agent:turn",
        "content_chunk": "ui:content:chunk",
        "content_start": "ui:content:start",
        "content_end": "ui:content:end",
        "human_input_requested": "ui:human:input_requested",
        "human_input_received": "ui:human:input_received",
        # New executor-specific events
        "node_start": "ui:graph:node_start",
        "node_complete": "ui:graph:node_complete",
        "node_error": "ui:graph:node_error",
        "edge_traversed": "ui:graph:edge_traversed",
        "task_start": "ui:task:start",
        "task_complete": "ui:task:complete",
        "execution_start": "ui:execution:start",
        "execution_complete": "ui:execution:complete",
        "execution_error": "ui:execution:error",
    }

    return event_mapping.get(event_type, f"ui:event:{event_type}")
