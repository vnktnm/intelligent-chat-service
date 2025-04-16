from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Body,
    Query,
    File,
    UploadFile,
    Form,
)
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from core import OpenAIService, get_openai_service
from utils import logger, export_graph_to_dot, export_graph_to_json
from utils.graph_yaml_loader import load_graph_from_yaml, load_graph_from_yaml_file
from core.database.mongo import MongoDBManager
import uuid
from datetime import datetime
import json
import yaml
import os

graph_router = APIRouter(prefix="/graph", tags=["Graph"])

# Registry to keep track of graph orchestrators
graph_registry = {}

# MongoDB collection name for graph executions
MONGO_GRAPH_COLLECTION = "graph_executions"


class GraphDefinition(BaseModel):
    """API schema for graph definition."""

    name: str
    description: str
    nodes: List[Dict[str, Any]]


class GraphExecution(BaseModel):
    """API schema for graph execution record."""

    graph_id: str
    execution_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    context: Dict[str, Any]
    nodes_total: Optional[int] = None
    nodes_completed: Optional[int] = None
    nodes_error: Optional[int] = None
    nodes_skipped: Optional[int] = None
    execution_order: Optional[List[str]] = None
    error_message: Optional[str] = None
    thread_id: Optional[str] = None


# Function to get MongoDB manager as a dependency
def get_mongodb_manager():
    mongodb = MongoDBManager()
    try:
        yield mongodb
    finally:
        mongodb.close()


@graph_router.post("/register")
async def register_graph(graph_def: GraphDefinition):
    """Register a graph orchestrator for later use."""
    graph_id = f"graph_{uuid.uuid4().hex[:8]}"
    graph_registry[graph_id] = {
        "definition": graph_def.dict(),
        "created_at": datetime.now().isoformat(),
        "status": "registered",
    }

    return {
        "graph_id": graph_id,
        "message": "Graph registered successfully",
    }


@graph_router.get("/list")
async def list_graphs():
    """List all registered graph orchestrators."""
    return {
        "graphs": [
            {
                "graph_id": graph_id,
                "name": graph_data["definition"]["name"],
                "created_at": graph_data["created_at"],
                "status": graph_data["status"],
            }
            for graph_id, graph_data in graph_registry.items()
        ]
    }


@graph_router.get("/{graph_id}")
async def get_graph(graph_id: str):
    """Get details about a specific graph."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    return graph_registry[graph_id]


@graph_router.get("/{graph_id}/visualize")
async def visualize_graph(
    graph_id: str, format: str = Query("json", enum=["json", "dot"])
):
    """Visualize a graph in the specified format."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph_data = graph_registry[graph_id]

    # Extract nodes and edges from definition
    nodes = []
    edges = []

    for node in graph_data["definition"]["nodes"]:
        nodes.append(
            {
                "id": node["id"],
                "label": node.get("name", node["id"]),
                "status": "pending",
                "metadata": node.get("metadata", {}),
            }
        )

        for dep_id in node.get("dependencies", []):
            edges.append({"from": dep_id, "to": node["id"]})

    # Return in requested format
    if format == "dot":
        dot_content = export_graph_to_dot(nodes, edges)
        return {"dot": dot_content}
    else:
        return {"graph": {"nodes": nodes, "edges": edges}}


@graph_router.delete("/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a registered graph."""
    if graph_id not in graph_registry:
        raise HTTPException(status_code=404, detail="Graph not found")

    del graph_registry[graph_id]
    return {"message": f"Graph {graph_id} deleted successfully"}


@graph_router.post("/from-yaml")
async def create_graph_from_yaml(
    yaml_content: str = Body(..., media_type="text/plain")
):
    """Create a graph orchestrator from YAML definition."""
    try:
        # Parse the YAML to validate it
        try:
            yaml_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid YAML format: {str(e)}"
            )

        # Create a unique ID for this graph
        graph_id = f"graph_{uuid.uuid4().hex[:8]}"

        # Store the original YAML
        graph_registry[graph_id] = {
            "definition": yaml_data,
            "created_at": datetime.now().isoformat(),
            "status": "registered",
            "source": "yaml",
            "yaml_content": yaml_content,
        }

        return {
            "graph_id": graph_id,
            "message": "Graph created from YAML successfully",
            "name": yaml_data.get("name", "Unnamed Graph"),
        }

    except Exception as e:
        logger.error(f"Error creating graph from YAML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create graph: {str(e)}")


@graph_router.post("/upload-yaml")
async def upload_yaml_graph(file: UploadFile = File(...)):
    """Upload a YAML file to create a graph orchestrator."""
    try:
        # Read the file content
        content = await file.read()
        yaml_content = content.decode("utf-8")

        # Parse the YAML to validate it
        try:
            yaml_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid YAML format: {str(e)}"
            )

        # Create a unique ID for this graph
        graph_id = f"graph_{uuid.uuid4().hex[:8]}"

        # Store the graph definition
        graph_registry[graph_id] = {
            "definition": yaml_data,
            "created_at": datetime.now().isoformat(),
            "status": "registered",
            "source": "yaml_upload",
            "filename": file.filename,
            "yaml_content": yaml_content,
        }

        return {
            "graph_id": graph_id,
            "message": f"Graph uploaded successfully from file {file.filename}",
            "name": yaml_data.get("name", "Unnamed Graph"),
        }

    except Exception as e:
        logger.error(f"Error uploading YAML graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload graph: {str(e)}")


@graph_router.get("/executions/list")
async def list_graph_executions(
    graph_id: Optional[str] = None,
    status: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """List all graph executions with optional filtering."""
    try:
        # Build the filter query
        filter_query = {}
        if graph_id:
            filter_query["graph_id"] = graph_id
        if status:
            filter_query["status"] = status
        if thread_id:
            filter_query["thread_id"] = thread_id

        # Query MongoDB for executions
        executions = await mongodb.find(
            MONGO_GRAPH_COLLECTION, filter_query, skip=skip, limit=limit
        )

        # Count total matching executions
        total_count = await mongodb.count(MONGO_GRAPH_COLLECTION, filter_query)

        return {
            "executions": executions,
            "total": total_count,
            "limit": limit,
            "skip": skip,
        }
    except Exception as e:
        logger.error(f"Error listing graph executions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list graph executions: {str(e)}"
        )


@graph_router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: str,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """Get details about a specific graph execution."""
    try:
        # Get execution from MongoDB
        execution = await mongodb.find_one(
            MONGO_GRAPH_COLLECTION, {"execution_id": execution_id}
        )

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution with ID {execution_id} not found"
            )

        return execution
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving graph execution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve graph execution: {str(e)}"
        )


@graph_router.get("/executions/thread/{thread_id}")
async def get_thread_executions(
    thread_id: str,
    limit: int = 50,
    skip: int = 0,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """Get all executions for a specific thread."""
    try:
        # Query MongoDB for executions by thread_id
        filter_query = {"thread_id": thread_id}

        executions = await mongodb.find(
            MONGO_GRAPH_COLLECTION, filter_query, skip=skip, limit=limit
        )

        # Count total executions for this thread
        total_count = await mongodb.count(MONGO_GRAPH_COLLECTION, filter_query)

        return {
            "thread_id": thread_id,
            "executions": executions,
            "total": total_count,
            "limit": limit,
            "skip": skip,
        }
    except Exception as e:
        logger.error(f"Error retrieving thread executions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve thread executions: {str(e)}"
        )


@graph_router.get("/executions/{execution_id}/visualize")
async def visualize_execution(
    execution_id: str,
    format: str = Query("json", enum=["json", "dot"]),
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """Visualize a graph execution with current execution state."""
    try:
        # Get execution from MongoDB
        execution = await mongodb.find_one(
            MONGO_GRAPH_COLLECTION, {"execution_id": execution_id}
        )

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution with ID {execution_id} not found"
            )

        # Get corresponding graph definition
        graph_id = execution.get("graph_id")
        if not graph_id or graph_id not in graph_registry:
            raise HTTPException(
                status_code=404,
                detail=f"Graph definition for execution {execution_id} not found",
            )

        graph_data = graph_registry[graph_id]

        # Extract nodes and edges from definition
        nodes = []
        edges = []

        # Add nodes with execution status
        for node in graph_data["definition"]["nodes"]:
            node_id = node["id"]
            node_status = "pending"  # Default status

            # Check if node exists in execution_order (completed)
            if (
                "execution_order" in execution
                and node_id in execution["execution_order"]
            ):
                node_status = "completed"

            # Check if node exists in error nodes
            if "error_nodes" in execution and node_id in execution["error_nodes"]:
                node_status = "error"

            # Check if node exists in skipped nodes
            if "skipped_nodes" in execution and node_id in execution["skipped_nodes"]:
                node_status = "skipped"

            nodes.append(
                {
                    "id": node_id,
                    "label": node.get("name", node_id),
                    "status": node_status,
                    "metadata": node.get("metadata", {}),
                }
            )

            # Add standard edges
            for dep_id in node.get("dependencies", []):
                edges.append({"from": dep_id, "to": node_id, "type": "static"})

        # Return in requested format
        if format == "dot":
            dot_content = export_graph_to_dot(nodes, edges)
            return {"dot": dot_content}
        else:
            return {
                "graph": {"nodes": nodes, "edges": edges},
                "execution": {
                    "id": execution["execution_id"],
                    "status": execution["status"],
                    "duration_ms": execution.get("duration_ms"),
                    "success": execution.get("success", False),
                    "nodes_total": execution.get("nodes_total", 0),
                    "nodes_completed": execution.get("nodes_completed", 0),
                    "nodes_error": execution.get("nodes_error", 0),
                    "nodes_skipped": execution.get("nodes_skipped", 0),
                    "execution_order": execution.get("execution_order", []),
                },
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error visualizing graph execution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to visualize graph execution: {str(e)}"
        )


@graph_router.delete("/executions/{execution_id}")
async def delete_execution(
    execution_id: str,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """Delete a graph execution record."""
    try:
        # Check if execution exists
        execution = await mongodb.find_one(
            MONGO_GRAPH_COLLECTION, {"execution_id": execution_id}
        )

        if not execution:
            raise HTTPException(
                status_code=404, detail=f"Execution with ID {execution_id} not found"
            )

        # Delete from MongoDB
        deleted_count = await mongodb.delete_one(
            MONGO_GRAPH_COLLECTION, {"execution_id": execution_id}
        )

        if deleted_count == 0:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete execution {execution_id}"
            )

        return {"message": f"Execution {execution_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting graph execution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete graph execution: {str(e)}"
        )


# Function to record graph execution start
async def record_execution_start(
    graph_id: str,
    context: Dict[str, Any],
    mongodb: MongoDBManager,
    thread_id: Optional[str] = None,
) -> str:
    """Record the start of a graph execution."""
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"

    execution_record = {
        "graph_id": graph_id,
        "execution_id": execution_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "context": context,
        "thread_id": thread_id,
    }

    await mongodb.insert(MONGO_GRAPH_COLLECTION, execution_record)
    return execution_id


# Function to record graph execution completion
async def record_execution_complete(
    execution_id: str,
    success: bool,
    stats: Dict[str, Any],
    error_message: Optional[str] = None,
    mongodb: MongoDBManager = None,
) -> bool:
    """Record the completion of a graph execution."""
    if not mongodb:
        mongodb = MongoDBManager()

    try:
        end_time = datetime.now().isoformat()

        # Build the update data
        update_data = {
            "$set": {
                "status": "completed" if success else "failed",
                "end_time": end_time,
                "success": success,
                "duration_ms": stats.get("duration_ms"),
                "nodes_total": stats.get("nodes_total"),
                "nodes_completed": stats.get("nodes_completed"),
                "nodes_error": stats.get("nodes_error"),
                "nodes_skipped": stats.get("nodes_skipped"),
                "execution_order": stats.get("execution_order", []),
            }
        }

        if error_message:
            update_data["$set"]["error_message"] = error_message

        # Update in MongoDB
        await mongodb.update_one(
            MONGO_GRAPH_COLLECTION, {"execution_id": execution_id}, update_data
        )

        return True
    except Exception as e:
        logger.error(f"Error recording graph execution completion: {str(e)}")
        return False
