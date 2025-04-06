from typing import Optional, Dict, Any, List, Union
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
import config
from utils import logger
from sentence_transformers import CrossEncoder


class QdrantManager:
    """Utility class to connect to Qdrant vector database"""

    def __init__(
        self,
        collection_name: Optional[str] = None,
    ):
        self.qdrant_host = config.QDRANT_HOST
        self.qdrant_port = config.QDRANT_PORT
        self.collection_name = collection_name
        self._client = None

    @property
    def client(self) -> QdrantClient:
        """Returns the Qdrant client, creating it if necessary"""
        if self._client is None:
            self._client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
        return self._client

    def is_connected(self) -> bool:
        """Check if connected to Qdrant"""
        try:
            if self._client is not None:
                # Try a simple operation to verify connection
                self._client.get_collections()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False

    def close(self) -> None:
        """Close the Qdrant client connection"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def create_collection(
        self, collection_name: str, vector_size: int, distance: str = "Cosine"
    ) -> bool:
        """
        Create a new collection in Qdrant

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            distance: Distance metric (Cosine, Euclid, Dot)

        Returns:
            bool: True if successful
        """
        try:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE),
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def create_collection_if_not_exists(
        self, collection_name: str, vector_size: int = None, distance: str = "Cosine"
    ) -> bool:
        """
        Create a new collection in Qdrant if it doesn't exist

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors (uses config default if None)
            distance: Distance metric (Cosine, Euclid, Dot)

        Returns:
            bool: True if successful or if collection already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            # If collection exists, return True
            if collection_name in collection_names:
                logger.info(f"Collection {collection_name} already exists")
                return True

            # Use default vector size from config if not provided
            if vector_size is None:
                vector_size = config.EMBEDDING_VECTOR_SIZE

            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE),
                ),
            )
            logger.info(
                f"Created new collection: {collection_name} with vector size {vector_size}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def add_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """
        Add points to the collection

        Args:
            collection_name: Name of the collection
            points: List of points with id, vector, and payload

        Returns:
            bool: True if successful
        """
        try:
            point_structs = [
                PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
                for point in points
            ]

            self.client.upsert(collection_name=collection_name, points=point_structs)
            return True
        except Exception as e:
            logger.error(f"Failed to add points: {e}")
            return False

    def rerank_with_cross_encoder(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model

        Args:
            query: The original search query text
            initial_results: Initial vector search results
            model_name: Name of the cross-encoder model to use

        Returns:
            List of reranked search results
        """
        try:
            # Initialize cross-encoder model
            cross_encoder = CrossEncoder(model_name)

            # Prepare pairs for reranking
            pairs = []
            for result in initial_results:
                # Try to extract document text from payload
                # Adjust the key based on your actual payload structure
                document_text = result.get("payload", {}).get("text", "")
                if not document_text and "content" in result.get("payload", {}):
                    document_text = result["payload"]["content"]

                pairs.append([query, document_text])

            # Get scores from cross-encoder
            if not pairs:
                return initial_results

            scores = cross_encoder.predict(pairs)

            # Add cross-encoder scores to results
            for idx, result in enumerate(initial_results):
                result["cross_encoder_score"] = float(scores[idx])

            # Sort by cross-encoder score (descending)
            reranked_results = sorted(
                initial_results,
                key=lambda x: x.get("cross_encoder_score", 0.0),
                reverse=True,
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fall back to the original results if reranking fails
            return initial_results

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_params: Optional[Dict[str, Any]] = None,
        reranker: Optional[callable] = None,
        query_text: Optional[str] = None,
        use_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection

        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            filter_params: Filter parameters
            reranker: Optional function to rerank results, should accept list of results and return reranked list
            query_text: Original query text (needed for cross-encoder reranking)
            use_cross_encoder: Whether to use cross-encoder for reranking
            cross_encoder_model: Name of the cross-encoder model to use

        Returns:
            List of search results
        """
        try:
            # Fetch more results if reranking will be applied
            search_limit = limit * 3 if (reranker or use_cross_encoder) else limit

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=search_limit,
                query_filter=filter_params,
            )

            # Convert results to a more usable format
            formatted_results = [
                {"id": res.id, "score": res.score, "payload": res.payload}
                for res in results
            ]

            # Apply cross-encoder reranking if requested
            if use_cross_encoder and query_text:
                formatted_results = self.rerank_with_cross_encoder(
                    query_text, formatted_results, cross_encoder_model
                )
                return formatted_results[:limit]

            # Apply custom reranking if provided
            if reranker:
                reranked_results = reranker(formatted_results)
                return reranked_results[:limit]

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
