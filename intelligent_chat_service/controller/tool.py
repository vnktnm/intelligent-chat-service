from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from core.database.vector import QdrantManager
from core.database.mongo import MongoDBManager
from utils import logger
import config
import uuid
from core.llm.openai import get_openai_service, OpenAIService

# Create router for tool endpoints
tool_router = APIRouter(prefix="/ai/tools", tags=["Tools"])


# Vector embedding model using OpenAI's embedding API
class EmbeddingModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_embeddings(self, text: str) -> List[float]:
        openai_service = get_openai_service()
        try:
            embedding = await openai_service.client.embeddings.create(
                model="text-embedding-3-small", input=text, encoding_format="float"
            )
            return embedding.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embeddings: {str(e)}"
            )


# Get embedding model as a dependency
def get_embedding_model():
    return EmbeddingModel.get_instance()


# Get Qdrant manager as a dependency
def get_qdrant_manager():
    qdrant = QdrantManager(collection_name=config.TOOL_COLLECTION_NAME)
    try:
        yield qdrant
    finally:
        qdrant.close()


# Get MongoDB manager as a dependency
def get_mongodb_manager():
    return MongoDBManager()


# Tool schema models
class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolCreate(BaseModel):
    name: str = Field(..., description="Unique name for the tool")
    description: str = Field(
        ..., description="Detailed description of what the tool does"
    )
    parameters: List[ToolParameter] = Field(
        default_factory=list, description="Parameters required by the tool"
    )
    endpoint: Optional[str] = Field(None, description="API endpoint for the tool")
    category: str = Field("uncategorized", description="Category of the tool")
    tags: List[str] = Field(default_factory=list, description="Tags for the tool")
    version: str = Field("1.0.0", description="Tool version")
    enabled: bool = Field(True, description="Whether the tool is enabled")


class ToolUpdate(BaseModel):
    description: Optional[str] = None
    parameters: Optional[List[ToolParameter]] = None
    endpoint: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    version: Optional[str] = None
    enabled: Optional[bool] = None


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    parameters: List[ToolParameter]
    endpoint: Optional[str] = None
    category: str
    tags: List[str]
    version: str
    enabled: bool


class ToolSearchResult(BaseModel):
    tools: List[ToolResponse]
    total: int


@tool_router.post("/", response_model=ToolResponse)
async def create_tool(
    tool: ToolCreate,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    Create a new tool with vector embeddings for semantic search
    """
    try:
        # Check if collection exists, if not create it
        collections = qdrant.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        # Generate tool ID
        tool_id = str(uuid.uuid4())

        # Prepare full payload for MongoDB
        tool_data = tool.dict()
        tool_data["id"] = tool_id

        # Create document for embedding - just the search-relevant fields
        embedding_text = f"{tool.name} {tool.description} {' '.join(tool.tags)}"
        for param in tool.parameters:
            embedding_text += f" {param.name} {param.description}"

        # Generate embedding using OpenAI
        vector = await embedding_model.get_embeddings(embedding_text)
        vector_size = len(vector)

        # Create collection if it doesn't exist
        if config.TOOL_COLLECTION_NAME not in collection_names:
            qdrant.create_collection(
                collection_name=config.TOOL_COLLECTION_NAME, vector_size=vector_size
            )
            logger.info(f"Created tool collection: {config.TOOL_COLLECTION_NAME}")

        # Create search payload with minimal data for Qdrant
        search_payload = {
            "id": tool_id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "tags": tool.tags,
            "enabled": tool.enabled,
        }

        # Add to vector database (Qdrant) for search
        point = {"id": tool_id, "vector": vector, "payload": search_payload}

        if not qdrant.add_points(config.TOOL_COLLECTION_NAME, [point]):
            raise HTTPException(
                status_code=500, detail="Failed to add tool to vector database"
            )

        # Store full data in MongoDB using the existing manager
        await mongodb.insert(config.MONGO_TOOL_COLLECTION_NAME, tool_data)

        logger.info(f"Created new tool: {tool.name} with ID: {tool_id}")
        return {**tool_data}

    except Exception as e:
        logger.error(f"Error creating tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {str(e)}")


@tool_router.get("/list", response_model=ToolSearchResult)
async def list_tools(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    enabled: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    List all tools with optional filtering
    """
    try:
        # Build MongoDB filter
        filter_query = {}

        if category:
            filter_query["category"] = category

        if tag:
            filter_query["tags"] = {"$in": [tag]}

        if enabled is not None:
            filter_query["enabled"] = enabled

        # Get total count for pagination
        total_count = await mongodb.count(
            config.MONGO_TOOL_COLLECTION_NAME, filter_query
        )

        # Get paginated results from MongoDB
        tools = await mongodb.find(
            config.MONGO_TOOL_COLLECTION_NAME,
            filter_query,
            skip=skip,
            limit=limit,
        )

        return {"tools": tools, "total": total_count}

    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@tool_router.post("/search", response_model=ToolSearchResult)
async def search_tools(
    query: str = Query(..., description="Search query for finding tools"),
    category: Optional[str] = None,
    enabled: Optional[bool] = True,
    limit: int = 10,
    rerank: bool = True,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    Search for tools using semantic search with optional reranking
    """
    try:
        # Create embedding for query using OpenAI
        query_vector = await embedding_model.get_embeddings(query)

        # Perform search without filters first
        results = qdrant.search(
            collection_name=config.TOOL_COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit * 3,  # Get more results than needed for filtering
            use_cross_encoder=rerank,
            query_text=query,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        # Filter results manually
        filtered_results = []
        tool_ids = []
        for result in results:
            payload = result["payload"]

            # Apply filtering manually
            include = True

            if category and payload.get("category") != category:
                include = False

            if enabled is not None and payload.get("enabled") != enabled:
                include = False

            if include:
                filtered_results.append(result)
                tool_ids.append(payload["id"])

        # Trim to required limit
        filtered_results = filtered_results[:limit]
        tool_ids = tool_ids[:limit]

        # Get complete tool data from MongoDB using the IDs we found via vector search
        tools = await mongodb.find(
            config.MONGO_TOOL_COLLECTION_NAME,
            {"id": {"$in": tool_ids}},
        )

        # Sort tools to match the order of results from Qdrant search
        tools_dict = {tool["id"]: tool for tool in tools}
        sorted_tools = [
            tools_dict.get(tool_id) for tool_id in tool_ids if tool_id in tools_dict
        ]

        # Add search score to tools
        for i, tool in enumerate(sorted_tools):
            if tool:
                # Find the matching result from filtered_results
                for result in filtered_results:
                    if result["payload"]["id"] == tool["id"]:
                        # Add the score from vector search
                        tool["search_score"] = result["score"]
                        # If reranked, also add that score
                        if rerank and "cross_encoder_score" in result:
                            tool["rerank_score"] = result["cross_encoder_score"]
                        break

        return {"tools": sorted_tools, "total": len(sorted_tools)}

    except Exception as e:
        logger.error(f"Error searching tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search tools: {str(e)}")


@tool_router.get("/tool/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    Get a tool by its ID from MongoDB
    """
    try:
        # Get tool from MongoDB
        tool_data = await mongodb.find_one(
            config.MONGO_TOOL_COLLECTION_NAME, {"id": tool_id}
        )

        if not tool_data:
            raise HTTPException(
                status_code=404, detail=f"Tool with ID {tool_id} not found"
            )

        return tool_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving tool: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve tool: {str(e)}"
        )


@tool_router.put("/tool/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    tool_update: ToolUpdate,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    Update an existing tool in both MongoDB and Qdrant
    """
    try:
        # First get the existing tool from MongoDB
        tool_data = await mongodb.find_one(
            config.MONGO_TOOL_COLLECTION_NAME, {"id": tool_id}
        )

        if not tool_data:
            raise HTTPException(
                status_code=404, detail=f"Tool with ID {tool_id} not found"
            )

        # Update the tool data
        update_data = tool_update.dict(exclude_unset=True)

        for key, value in update_data.items():
            if value is not None:
                tool_data[key] = value

        # Update in MongoDB
        await mongodb.update_one(
            config.MONGO_TOOL_COLLECTION_NAME, {"id": tool_id}, {"$set": tool_data}
        )

        # Only update Qdrant if any of the searchable fields were changed
        should_update_qdrant = any(
            field in update_data
            for field in ["name", "description", "category", "tags", "enabled"]
        )

        if should_update_qdrant:
            # Create search payload with minimal data for Qdrant
            search_payload = {
                "id": tool_id,
                "name": tool_data["name"],
                "description": tool_data["description"],
                "category": tool_data["category"],
                "tags": tool_data["tags"],
                "enabled": tool_data["enabled"],
            }

            # Re-create embedding with updated data
            embedding_text = f"{tool_data['name']} {tool_data['description']} {' '.join(tool_data['tags'])}"
            for param in tool_data["parameters"]:
                embedding_text += f" {param['name']} {param['description']}"

            # Generate embedding using OpenAI
            vector = await embedding_model.get_embeddings(embedding_text)

            # Update in vector database
            point = {"id": tool_id, "vector": vector, "payload": search_payload}

            if not qdrant.add_points(config.TOOL_COLLECTION_NAME, [point]):
                raise HTTPException(
                    status_code=500, detail="Failed to update tool in vector database"
                )

        logger.info(f"Updated tool: {tool_data['name']} with ID: {tool_id}")
        return tool_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update tool: {str(e)}")


@tool_router.delete("/tool/{tool_id}")
async def delete_tool(
    tool_id: str,
    qdrant: QdrantManager = Depends(get_qdrant_manager),
    mongodb: MongoDBManager = Depends(get_mongodb_manager),
):
    """
    Delete a tool by its ID from both MongoDB and Qdrant
    """
    try:
        # Check if tool exists in MongoDB
        tool_data = await mongodb.find_one(
            config.MONGO_TOOL_COLLECTION_NAME, {"id": tool_id}
        )

        if not tool_data:
            raise HTTPException(
                status_code=404, detail=f"Tool with ID {tool_id} not found"
            )

        # Delete from MongoDB
        await mongodb.delete_one(config.MONGO_TOOL_COLLECTION_NAME, {"id": tool_id})

        # Delete from Qdrant vector database
        try:
            qdrant.client.delete(
                collection_name=config.TOOL_COLLECTION_NAME, points_selector=[tool_id]
            )
        except Exception as e:
            # Log error but don't fail if vector deletion fails (tool is already removed from MongoDB)
            logger.warning(f"Error deleting tool from vector database: {str(e)}")

        logger.info(f"Deleted tool with ID: {tool_id}")
        return {"status": "success", "message": f"Tool {tool_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")
