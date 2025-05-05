from typing import List, Dict, Any, Optional, Union, Tuple
import os
import logging
from dotenv import load_dotenv
import numpy as np
from pydantic import BaseModel, Field

# Vector operations
from fastembed import TextEmbedding, SparseTextEmbedding, SparseEmbedding
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models

# FastAPI imports
from fastapi import FastAPI, Query, HTTPException
import uvicorn

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SearchResult(BaseModel):
    """Model for search results"""

    document_name: str
    section_title: str
    page_num: int
    page_end: int
    text: str
    source_file: str
    score: float
    ingestion_timestamp: Optional[str] = None


class QdrantRetrievalTool:
    """Tool for retrieving information from Qdrant vector database"""

    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION", "document_collection")

        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        # Cache directory for models
        self.cache_dir = os.getenv("FASTEMBED_CACHE_DIR", "models_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize models lazily (they'll be loaded only when needed)
        self._dense_model = None
        self._sparse_model = None
        self._cross_encoder = None

        # Model configurations
        self.dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.sparse_model_name = "Qdrant/bm25"
        self.cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Verify collection exists
        self._verify_collection()

    def _verify_collection(self):
        """Verify that the collection exists in Qdrant"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Qdrant"
            )

        # Get collection info to confirm it has the right vector configuration
        collection_info = self.qdrant_client.get_collection(self.collection_name)

        # Verify the collection has required vector configurations
        if "dense" not in collection_info.config.params.vectors:
            raise ValueError("Collection doesn't have 'dense' vector configuration")

        # Fix: Access sparse vectors correctly from the collection configuration
        try:
            if hasattr(collection_info.config, "params") and hasattr(
                collection_info.config.params, "sparse_vectors"
            ):
                if "sparse" not in collection_info.config.params.sparse_vectors:
                    raise ValueError(
                        "Collection doesn't have 'sparse' vector configuration"
                    )
            else:
                logger.warning(
                    "Collection info structure doesn't have expected sparse_vectors attribute path. Skipping sparse vector check."
                )
        except Exception as e:
            logger.warning(
                f"Error checking sparse vectors configuration: {str(e)}. Continuing anyway."
            )

    @property
    def dense_model(self):
        """Lazy-load the dense embedding model"""
        if self._dense_model is None:
            logger.info(f"Loading dense embedding model: {self.dense_model_name}")
            self._dense_model = TextEmbedding(
                model_name=self.dense_model_name, cache_dir=self.cache_dir
            )
        return self._dense_model

    @property
    def sparse_model(self):
        """Lazy-load the sparse embedding model"""
        if self._sparse_model is None:
            logger.info(f"Loading sparse embedding model: {self.sparse_model_name}")
            self._sparse_model = SparseTextEmbedding(
                model_name=self.sparse_model_name, cache_dir=self.cache_dir
            )
        return self._sparse_model

    @property
    def cross_encoder(self):
        """Lazy-load the cross-encoder model"""
        if self._cross_encoder is None:
            logger.info(f"Loading cross-encoder model: {self.cross_encoder_name}")
            self._cross_encoder = CrossEncoder(self.cross_encoder_name)
        return self._cross_encoder

    def _generate_dense_embedding(self, query: str) -> List[float]:
        """Generate dense embedding for a query"""
        embeddings = list(self.dense_model.embed([query]))
        return embeddings[0].tolist()

    def _generate_sparse_embedding(self, query: str) -> SparseEmbedding:
        """Generate sparse embedding for a query"""
        embeddings = list(self.sparse_model.embed([query]))
        return embeddings[0]

    def _format_search_results(
        self, scored_points: List[Tuple[models.Record, float]]
    ) -> List[SearchResult]:
        """Format search results into a standardized format"""
        results = []
        for record, score in scored_points:
            payload = record.payload
            result = SearchResult(
                document_name=payload.get("document_name", "Unknown"),
                section_title=payload.get("section_title", "Unknown"),
                page_num=payload.get("page_num", 0),
                page_end=payload.get("page_end", 0),
                text=payload.get("text", ""),
                source_file=payload.get("source_file", "Unknown"),
                ingestion_timestamp=payload.get("ingestion_timestamp", None),
                score=score,
            )
            results.append(result)
        return results

    def _rerank_with_cross_encoder(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results

        # Prepare pairs for cross-encoder
        pairs = [[query, result.text] for result in results]

        # Get scores from cross-encoder
        cross_scores = self.cross_encoder.predict(pairs)

        # Create new list with cross-encoder scores
        scored_results = [
            (result, float(score)) for result, score in zip(results, cross_scores)
        ]

        # Sort by cross-encoder score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Take top_k and update scores
        reranked_results = []
        for result, score in scored_results[:top_k]:
            result.score = score  # Update the score with cross-encoder score
            reranked_results.append(result)

        return reranked_results

    async def semantic_search(
        self, query: str, limit: int = 5, score_threshold: float = 0.6
    ) -> List[SearchResult]:
        """
        Perform semantic search using dense vectors

        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of search results
        """
        dense_vector = self._generate_dense_embedding(query)

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_vector),
            limit=limit,
            score_threshold=score_threshold,
        )

        # Format results
        return self._format_search_results([(hit, hit.score) for hit in search_results])

    async def keyword_search(
        self, query: str, limit: int = 5, score_threshold: float = 0.2
    ) -> List[SearchResult]:
        """
        Perform keyword search using sparse vectors

        Args:
            query: The search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of search results
        """
        sparse_vector = self._generate_sparse_embedding(query)

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=(
                "sparse",
                models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
            ),
            limit=limit,
            score_threshold=score_threshold,
        )

        # Format results
        return self._format_search_results([(hit, hit.score) for hit in search_results])

    async def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rerank: bool = False,
        rerank_top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform hybrid search using both dense and sparse vectors

        Args:
            query: The search query
            limit: Maximum number of results to return
            dense_weight: Weight for dense vector search (0 to 1)
            sparse_weight: Weight for sparse vector search (0 to 1)
            rerank: Whether to rerank using cross-encoder
            rerank_top_k: Number of top results to keep after reranking

        Returns:
            List of search results
        """
        # Generate both embeddings
        dense_vector = self._generate_dense_embedding(query)
        sparse_vector = self._generate_sparse_embedding(query)

        try:
            # Simplify approach - just use semantic search first
            logger.info("Performing dense vector search first")
            dense_results = await self.semantic_search(
                query=query,
                limit=limit if not rerank else max(limit, rerank_top_k),
                score_threshold=0.0,  # No threshold to get more results for hybrid search
            )

            # Then use sparse search
            logger.info("Performing sparse vector search")
            sparse_results = await self.keyword_search(
                query=query,
                limit=limit if not rerank else max(limit, rerank_top_k),
                score_threshold=0.0,  # No threshold to get more results for hybrid search
            )

            # Combine results with weighting
            combined_results = {}

            # Add dense results with weight
            for result in dense_results:
                combined_results[result.text] = {
                    "result": result,
                    "score": result.score * dense_weight,
                }

            # Add or update with sparse results
            for result in sparse_results:
                if result.text in combined_results:
                    # Result exists from dense search, combine scores
                    combined_results[result.text]["score"] += (
                        result.score * sparse_weight
                    )
                else:
                    # New result from sparse search
                    combined_results[result.text] = {
                        "result": result,
                        "score": result.score * sparse_weight,
                    }

            # Convert back to list and update scores
            weighted_results = []
            for text, data in combined_results.items():
                result = data["result"]
                result.score = data["score"]  # Update with combined score
                weighted_results.append(result)

            # Sort by combined score
            weighted_results.sort(key=lambda x: x.score, reverse=True)

            # Limit results
            weighted_results = weighted_results[: limit if not rerank else rerank_top_k]

            # Rerank if requested
            if rerank and weighted_results:
                final_results = self._rerank_with_cross_encoder(
                    query=query,
                    results=weighted_results,
                    top_k=min(limit, len(weighted_results)),
                )
                return final_results

            return weighted_results[:limit]

        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            # Fallback to just dense search if anything fails
            logger.info("Falling back to dense search only")
            dense_results = await self.semantic_search(
                query=query,
                limit=limit,
                score_threshold=0.0,  # No threshold to get more results
            )
            return dense_results


# FastAPI Models for the API
class SearchQuery(BaseModel):
    """Model for search queries"""

    query: str
    collection_name: Optional[str] = None


class HybridSearchQuery(SearchQuery):
    """Model for hybrid search queries - simplified to only require query and collection"""

    pass  # No additional fields


# Initialize FastAPI app
app = FastAPI(
    title="Qdrant Retrieval API",
    description="API for retrieving information from Qdrant vector database using semantic, keyword, and hybrid search",
    version="1.0.0",
)

# Initialize retrieval tool as a global variable (will be lazy-loaded)
retrieval_tool = QdrantRetrievalTool()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Qdrant Retrieval API is running"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Check if Qdrant is responsive
        retrieval_tool.qdrant_client.get_collections()
        return {
            "status": "healthy",
            "message": "Service is running and connected to Qdrant",
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/search/semantic", response_model=List[SearchResult])
async def api_semantic_search(query: SearchQuery):
    """
    Perform semantic search using dense vectors
    """
    try:
        # Use collection name from query if provided, otherwise use default
        collection_name = query.collection_name or retrieval_tool.collection_name

        # Set optimal defaults
        limit = 5
        score_threshold = 0.6

        # Override tool's collection name temporarily if needed
        original_collection = retrieval_tool.collection_name
        if collection_name != original_collection:
            retrieval_tool.collection_name = collection_name

        try:
            results = await retrieval_tool.semantic_search(
                query=query.query, limit=limit, score_threshold=score_threshold
            )
        finally:
            # Restore original collection name
            if collection_name != original_collection:
                retrieval_tool.collection_name = original_collection

        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error performing semantic search: {str(e)}"
        )


@app.post("/search/keyword", response_model=List[SearchResult])
async def api_keyword_search(query: SearchQuery):
    """
    Perform keyword search using sparse vectors
    """
    try:
        # Use collection name from query if provided, otherwise use default
        collection_name = query.collection_name or retrieval_tool.collection_name

        # Set optimal defaults
        limit = 5
        score_threshold = 0.2

        # Override tool's collection name temporarily if needed
        original_collection = retrieval_tool.collection_name
        if collection_name != original_collection:
            retrieval_tool.collection_name = collection_name

        try:
            results = await retrieval_tool.keyword_search(
                query=query.query, limit=limit, score_threshold=score_threshold
            )
        finally:
            # Restore original collection name
            if collection_name != original_collection:
                retrieval_tool.collection_name = original_collection

        return results
    except Exception as e:
        logger.error(f"Error in keyword search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error performing keyword search: {str(e)}"
        )


@app.post("/search/hybrid", response_model=List[SearchResult])
async def api_hybrid_search(query: HybridSearchQuery):
    """
    Perform hybrid search using both dense and sparse vectors with cross-encoder reranking

    Uses default optimized parameters for best results
    """
    try:
        # Use collection name from query if provided, otherwise use default
        collection_name = query.collection_name or retrieval_tool.collection_name

        # Set optimal defaults
        limit = 5
        dense_weight = 0.5
        sparse_weight = 0.5
        rerank = True
        rerank_top_k = 10

        # Override tool's collection name temporarily if needed
        original_collection = retrieval_tool.collection_name
        if collection_name != original_collection:
            retrieval_tool.collection_name = collection_name

        try:
            results = await retrieval_tool.hybrid_search(
                query=query.query,
                limit=limit,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                rerank=rerank,
                rerank_top_k=rerank_top_k,
            )
        finally:
            # Restore original collection name
            if collection_name != original_collection:
                retrieval_tool.collection_name = original_collection

        return results
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error performing hybrid search: {str(e)}"
        )


# Run the API server if this file is executed directly
if __name__ == "__main__":
    port = 8001
    host = os.getenv("API_HOST", "0.0.0.0")
    logger.info(f"Starting Qdrant retrieval API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
