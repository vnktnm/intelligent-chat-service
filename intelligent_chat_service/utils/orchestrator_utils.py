from orchestrator import Orchestrator, IdiscoveryGraphOrchestrator
from schema import ChatRequest


def get_orchestrator(request: ChatRequest) -> Orchestrator:
    orchestrators = {"idiscovery_orchestrator": IdiscoveryGraphOrchestrator}

    if request.workflow_name not in orchestrators:
        raise ValueError(
            f"Orchestrator {request.workflow_name} not found. Available orchestrators: {list(orchestrators.keys())}"
        )

    return orchestrators[request.workflow_name](request)


def standardize_event_type(event_type: str) -> str:
    event_mapping = {
        "orchestrator_start": "ui:orchestrator:start",
        "orchestrator_progress": "ui:orchestrator:progress",
        "orchestrator_complete": "ui:orchestrator:complete",
        "step_update": "ui:step:update",
        "step_complete": "ui:step:complete",
        "content_chunk": "ui:content:chunk",
        "content_start": "ui:content:start",
        "content_end": "ui:content:end",
        "human_input_requested": "ui:human:input_requested",
        "human_input_received": "ui:human:input_received",
        "node_started": "ui:node:started",
        "node_completed": "ui:node:completed",
        "node_error": "ui:node:error",
        "node_skipped": "ui:node:skipped",
        "orchestrator_error": "ui:orchestrator:error",
        "conditional_edges_activated": "ui:node:conditional_edges_activated",
    }

    return event_mapping.get(event_type, f"ui:event:{event_type}")
