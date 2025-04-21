from fastapi import APIRouter, Depends, HTTPException, Request, Response
from typing import Dict, Any, List, Optional
from utils import logger, get_orchestrator
from schema import ChatRequest
from orchestrator.graph_orchestrator import GraphOrchestrator
import json
from fastapi.responses import HTMLResponse
from agents.executor_agent import ExecutorAgent

graph_router = APIRouter(prefix="/graph", tags=["Graph Orchestration"])


@graph_router.get("/orchestrators")
async def list_orchestrators():
    """List all available orchestrators."""
    try:
        # This is a placeholder. In a real implementation, you would dynamically
        # discover all available orchestrators from your system.
        available_orchestrators = [
            "idiscovery_orchestrator"
        ]  # Add more as they become available

        return {"status": "success", "orchestrators": available_orchestrators}
    except Exception as e:
        logger.error(f"Error listing orchestrators: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list orchestrators: {str(e)}"
        )


@graph_router.get("/orchestrator/{workflow_name}")
async def get_orchestrator_details(workflow_name: str):
    """Get details of a specific orchestrator."""
    try:
        # Create a dummy request just to get the orchestrator
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            return {
                "status": "warning",
                "message": f"Orchestrator {workflow_name} is not a GraphOrchestrator",
                "details": (
                    orchestrator.to_dict() if hasattr(orchestrator, "to_dict") else {}
                ),
            }

        return {"status": "success", "orchestrator": orchestrator.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting orchestrator details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get orchestrator details: {str(e)}"
        )


@graph_router.get("/orchestrator/{workflow_name}/dot")
async def get_orchestrator_graphviz(workflow_name: str):
    """Get GraphViz DOT representation of the orchestrator graph."""
    try:
        # Create a dummy request just to get the orchestrator
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        dot_graph = orchestrator.export_graphviz()

        return {"status": "success", "dot_graph": dot_graph}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting orchestrator graphviz: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get orchestrator graphviz: {str(e)}"
        )


@graph_router.get("/orchestrator/{workflow_name}/html", response_class=HTMLResponse)
async def get_orchestrator_html(workflow_name: str):
    """Get HTML visualization of the orchestrator graph."""
    try:
        # Create a dummy request just to get the orchestrator
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        html = orchestrator.export_html()

        return HTMLResponse(content=html)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting orchestrator HTML: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get orchestrator HTML: {str(e)}"
        )


@graph_router.get("/executions/{workflow_name}")
async def get_execution_history(workflow_name: str):
    """Get execution history for a specific orchestrator."""
    try:
        # Create a dummy request just to get the orchestrator
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        executions = orchestrator.get_execution_history()

        return {"status": "success", "executions": executions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting execution history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get execution history: {str(e)}"
        )


@graph_router.get("/execution/{workflow_name}/{execution_id}")
async def get_execution_details(workflow_name: str, execution_id: str):
    """Get details of a specific execution."""
    try:
        # Create a dummy request just to get the orchestrator
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        execution = orchestrator.get_execution(execution_id)

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution {execution_id} not found"
            )

        return {"status": "success", "execution": execution}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting execution details: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get execution details: {str(e)}"
        )


@graph_router.get(
    "/execution/{workflow_name}/{execution_id}/html", response_class=HTMLResponse
)
async def get_execution_html(workflow_name: str, execution_id: str):
    """Get HTML visualization of a specific execution."""
    try:
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        execution = orchestrator.get_execution(execution_id)

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution {execution_id} not found"
            )

        # Create a visualization of the execution path
        viz_html = await _create_execution_visualization(orchestrator, execution)

        return HTMLResponse(content=viz_html)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting execution HTML: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get execution HTML: {str(e)}"
        )


@graph_router.get(
    "/agent/executor/{workflow_name}/current", response_class=HTMLResponse
)
async def get_executor_current_subgraph(workflow_name: str):
    """Get current subgraph from the executor agent if available."""
    try:
        dummy_request = ChatRequest(
            workflow_name=workflow_name, user_input="", selected_sources=[]
        )

        orchestrator = get_orchestrator(dummy_request)

        if not isinstance(orchestrator, GraphOrchestrator):
            raise HTTPException(
                status_code=400,
                detail=f"Orchestrator {workflow_name} is not a GraphOrchestrator",
            )

        # Find the executor agent node
        executor_agent = None
        for node_id, step in orchestrator.steps_map.items():
            if isinstance(step, ExecutorAgent):
                executor_agent = step
                break

        if not executor_agent or not executor_agent.current_subgraph:
            raise HTTPException(
                status_code=404, detail="No active executor subgraph found"
            )

        # Return the HTML visualization of the subgraph
        html = executor_agent.current_subgraph.export_html()

        return HTMLResponse(content=html)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Orchestrator not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting executor subgraph: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get executor subgraph: {str(e)}"
        )


async def _create_execution_visualization(
    orchestrator: GraphOrchestrator, execution: Dict[str, Any]
) -> str:
    """Create HTML visualization of an execution path."""
    # Extract visited nodes and traversed edges
    visited_nodes = execution.get("nodes_visited", [])
    traversed_edges = execution.get("edges_traversed", [])

    # Create vis.js visualization
    nodes_data = []
    for node_id in orchestrator.graph.nodes:
        step = orchestrator.steps_map[node_id]
        node_type = step.__class__.__name__
        color = orchestrator._get_node_color(node_type)

        # Highlight visited nodes
        if node_id in visited_nodes:
            color = "#FFA500"  # Orange for visited nodes

        nodes_data.append(
            {
                "id": node_id,
                "label": f"{node_id}\n({node_type})",
                "title": step.description,
                "color": color,
            }
        )

    edges_data = []
    for u, v, data in orchestrator.graph.edges(data=True):
        label = data.get("label", "")
        has_condition = data.get("condition") is not None

        edge = {
            "from": u,
            "to": v,
            "label": label,
            "arrows": "to",
        }

        # Highlight traversed edges
        if (u, v) in traversed_edges:
            edge["color"] = {"color": "green", "opacity": 1.0}
            edge["width"] = 3
        elif has_condition:
            edge["dashes"] = True
            edge["color"] = {"color": "blue", "opacity": 0.6}

        edges_data.append(edge)

    # Create the HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Execution Visualization</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            #mynetwork {
                width: 800px;
                height: 600px;
                border: 1px solid lightgray;
            }
            .execution-details {
                margin-top: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h2>Execution Visualization</h2>
        <div class="execution-details">
            <p><strong>Execution ID:</strong> %s</p>
            <p><strong>Status:</strong> %s</p>
            <p><strong>Started:</strong> %s</p>
            <p><strong>Completed:</strong> %s</p>
        </div>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            const nodes = new vis.DataSet(%s);
            const edges = new vis.DataSet(%s);
            
            const container = document.getElementById('mynetwork');
            const data = {
                nodes: nodes,
                edges: edges
            };
            const options = {
                nodes: {
                    shape: 'box',
                    font: {
                        size: 14
                    }
                },
                edges: {
                    font: {
                        size: 12
                    }
                },
                physics: {
                    enabled: true,
                    hierarchicalRepulsion: {
                        centralGravity: 0.0,
                        springLength: 100,
                        springConstant: 0.01,
                        nodeDistance: 120
                    },
                    solver: 'hierarchicalRepulsion'
                },
                layout: {
                    hierarchical: {
                        direction: 'LR',
                        sortMethod: 'directed'
                    }
                }
            };
            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """ % (
        execution.get("id", "Unknown"),
        execution.get("status", "Unknown"),
        execution.get("started_at", "Unknown"),
        execution.get("completed_at", "Unknown"),
        json.dumps(nodes_data),
        json.dumps(edges_data),
    )

    return html
