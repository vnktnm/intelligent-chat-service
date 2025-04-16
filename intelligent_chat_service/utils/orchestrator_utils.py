from orchestrator import Orchestrator, IDiscoveryOrchestrator
from orchestrator.graph_basic_orchestrator import GraphBasicOrchestrator
from typing import Optional
from schema import ChatRequest
from utils.graph_yaml_loader import load_graph_from_yaml


def get_orchestrator(request: ChatRequest) -> Orchestrator:
    orchestrators = {
        "idiscovery_orchestrator": IDiscoveryOrchestrator,
        "graph_basic_orchestrator": GraphBasicOrchestrator,
    }

    # Check if this is a YAML-defined graph request
    if request.workflow_name.startswith("yaml:"):
        yaml_content = request.workflow_name[5:]  # Remove the "yaml:" prefix
        yaml_orchestrator = load_graph_from_yaml(yaml_content)
        if yaml_orchestrator:
            return yaml_orchestrator
        else:
            raise ValueError(f"Failed to load YAML-defined orchestrator")

    # Check if this is a registered graph ID
    elif request.workflow_name.startswith("graph:"):
        from controller.graph_controller import graph_registry

        graph_id = request.workflow_name[6:]  # Remove the "graph:" prefix
        if graph_id in graph_registry:
            graph_data = graph_registry[graph_id]

            # If this is a YAML-sourced graph
            if (
                graph_data.get("source") in ["yaml", "yaml_upload"]
                and "yaml_content" in graph_data
            ):
                yaml_orchestrator = load_graph_from_yaml(graph_data["yaml_content"])
                if yaml_orchestrator:
                    return yaml_orchestrator

            raise ValueError(
                f"Graph {graph_id} found but could not be loaded as an orchestrator"
            )
        else:
            raise ValueError(f"Graph {graph_id} not found in registry")

    # Regular workflow
    if request.workflow_name not in orchestrators:
        raise ValueError(
            f"Workflow {request.workflow_name} not found. Available workflows: {list(orchestrators.keys())}"
        )

    return orchestrators[request.workflow_name](request)


def standardize_event_type(event_type: str) -> str:
    """Convert internal event types to UI friendly event types with proper namespacing"""
    event_mapping = {
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
        "node_started": "ui:node:started",
        "node_completed": "ui:node:completed",
        "node_error": "ui:node:error",
        "node_skipped": "ui:node:skipped",
        "orchestrator_error": "ui:orchestrator:error",
    }

    return event_mapping.get(event_type, f"ui:event:{event_type}")
