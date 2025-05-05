import os
import argparse
import logging
from typing import List, Dict, Any, Tuple
import uuid
from pathlib import Path
from dotenv import load_dotenv

# PDF processing
import fitz  # PyMuPDF
import re
from tqdm import tqdm

# Vector operations
import numpy as np
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    SparseEmbedding,
)  # Import SparseEmbedding for type hint
from qdrant_client import QdrantClient
from qdrant_client.http import models
import concurrent.futures
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# --- Worker Functions ---
# These functions run in separate processes and initialize models there.


def _process_dense_batch_worker(
    model_name: str, cache_dir: str, batch_texts: List[str]
) -> List[List[float]]:
    """Worker function to process a batch for dense embeddings."""
    # Initialize model inside the worker process
    dense_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    embeddings_generator = dense_model.embed(batch_texts)
    return [embedding.tolist() for embedding in embeddings_generator]


def _process_sparse_batch_worker(
    model_name: str, cache_dir: str, batch_texts: List[str]
) -> List[SparseEmbedding]:
    """Worker function to process a batch for sparse embeddings."""
    # Initialize model inside the worker process
    sparse_model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
    sparse_embeddings_generator = sparse_model.embed(batch_texts)
    return list(sparse_embeddings_generator)


# --- PDFIngester Class ---


class PDFIngester:
    def __init__(self):
        # Set up Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )

        # Cache directory for models
        self.cache_dir = os.getenv(
            "FASTEMBED_CACHE_DIR", "models_cache"
        )  # Store cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory for models: {self.cache_dir}")

        # Store model names and dimensions, but DO NOT initialize models here
        self.dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.dense_dim = 384
        logger.info(
            f"Using dense model: {self.dense_model_name} with dimension {self.dense_dim}"
        )

        self.sparse_model_name = "Qdrant/bm25"
        logger.info(f"Using sparse model: {self.sparse_model_name}")
        # Removed model initializations:
        # self.dense_model = TextEmbedding(...)
        # self.sparse_model = SparseTextEmbedding(...)

        # Collection name for Qdrant
        self.collection_name = os.getenv("QDRANT_COLLECTION", "document_collection")

        # Ensure collection exists
        self._init_collection()

    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection '{self.collection_name}'")

            # Create collection with hybrid search capability using FastEmbed dimensions
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_dim,  # Dimension from FastEmbed dense model
                        distance=models.Distance.COSINE,
                    ),
                    # Sparse vectors are defined in sparse_vectors_config
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                        # Dimension/vocab size is implicit for sparse vectors in Qdrant
                    )
                },
            )

            # Create payload index for efficient filtering (remains the same)
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_num",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="section_title",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF, grouping by section"""
        logger.info(f"Extracting text from {pdf_path}")

        doc = fitz.open(pdf_path)
        document_name = Path(pdf_path).stem

        # Store sections with their content
        sections = []
        current_section = {
            "title": "Introduction",  # Default first section title
            "content": "",
            "page_start": 0,
            "page_end": 0,
        }

        # Function to detect section headers
        def is_section_header(text: str) -> bool:
            # Simple heuristic: all caps, short, ends with no punctuation
            return (
                len(text.strip()) < 50
                and text.strip().isupper()
                and not text.strip().endswith(".")
            )

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            lines = page_text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if is_section_header(line):
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        current_section["page_end"] = page_num
                        sections.append(current_section)

                    # Start new section
                    current_section = {
                        "title": line.strip(),
                        "content": "",
                        "page_start": page_num,
                        "page_end": page_num,
                    }
                else:
                    current_section["content"] += line + " "

            # Add page break for readability
            current_section["content"] += "\n\n"

        # Add the last section
        if current_section["content"].strip():
            current_section["page_end"] = len(doc) - 1
            sections.append(current_section)

        # Create chunks from sections
        chunks = []
        for section in sections:
            # Create smaller chunks if section is too large
            max_chunk_size = 1000  # characters
            content = section["content"]

            if len(content) > max_chunk_size:
                # Split by paragraphs first
                paragraphs = re.split(r"\n\s*\n", content)
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        # Save current chunk
                        chunks.append(
                            {
                                "document_name": document_name,
                                "section_title": section["title"],
                                "page_num": section["page_start"],
                                "page_end": section["page_end"],
                                "text": current_chunk.strip(),
                            }
                        )
                        current_chunk = para + "\n\n"

                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(
                        {
                            "document_name": document_name,
                            "section_title": section["title"],
                            "page_num": section["page_start"],
                            "page_end": section["page_end"],
                            "text": current_chunk.strip(),
                        }
                    )
            else:
                chunks.append(
                    {
                        "document_name": document_name,
                        "section_title": section["title"],
                        "page_num": section["page_start"],
                        "page_end": section["page_end"],
                        "text": content.strip(),
                    }
                )

        return chunks

    def ingest_pdf(self, pdf_path: str) -> None:
        """Main function to ingest a PDF file"""
        try:
            # Extract text chunks from PDF
            chunks = self.extract_text_from_pdf(pdf_path)

            if not chunks:
                logger.warning(f"No content extracted from {pdf_path}")
                return

            # Extract text for embedding
            texts = [chunk["text"] for chunk in chunks]

            # Prepare batches for parallel processing
            # Adjust batch sizes based on available memory/CPU
            dense_batch_size = 32
            sparse_batch_size = 32

            dense_batches = [
                texts[i : i + dense_batch_size]
                for i in range(0, len(texts), dense_batch_size)
            ]
            sparse_batches = [
                texts[i : i + sparse_batch_size]
                for i in range(0, len(texts), sparse_batch_size)
            ]

            # Run dense and sparse embedding generation in parallel
            dense_vectors_list = []
            sparse_vectors_list = []

            # Using ProcessPoolExecutor for CPU-bound FastEmbed operations
            with concurrent.futures.ProcessPoolExecutor() as executor:
                logger.info(
                    f"Submitting {len(dense_batches)} dense batches and {len(sparse_batches)} sparse batches to ProcessPoolExecutor."
                )

                # Submit tasks using worker functions
                dense_futures = [
                    executor.submit(
                        _process_dense_batch_worker,
                        self.dense_model_name,
                        self.cache_dir,
                        batch,
                    )
                    for batch in dense_batches
                ]
                sparse_futures = [
                    executor.submit(
                        _process_sparse_batch_worker,
                        self.sparse_model_name,
                        self.cache_dir,
                        batch,
                    )
                    for batch in sparse_batches
                ]

                # Collect dense results as they complete
                logger.info("Waiting for dense embedding results...")
                for future in tqdm(
                    concurrent.futures.as_completed(dense_futures),
                    total=len(dense_futures),
                    desc="Dense Embeddings",
                ):
                    try:
                        result = future.result()
                        dense_vectors_list.extend(result)
                    except Exception as e:
                        logger.error(f"Error in dense vector generation task: {str(e)}")
                        # Potentially cancel other futures or handle error appropriately
                        raise

                # Collect sparse results as they complete
                logger.info("Waiting for sparse embedding results...")
                for future in tqdm(
                    concurrent.futures.as_completed(sparse_futures),
                    total=len(sparse_futures),
                    desc="Sparse Embeddings",
                ):
                    try:
                        result = future.result()
                        sparse_vectors_list.extend(result)
                    except Exception as e:
                        logger.error(
                            f"Error in sparse vector generation task: {str(e)}"
                        )
                        # Potentially cancel other futures or handle error appropriately
                        raise

            # Ensure results are correctly ordered (ProcessPoolExecutor + as_completed doesn't guarantee order)
            # We need to re-associate results with original chunks if order matters strictly,
            # but since we extend lists from completed futures, the final list order might be mixed.
            # A safer approach is to submit tasks with identifiers or process sequentially if order is critical.
            # However, for batch upsert, the order within the final lists doesn't matter as much as
            # ensuring each vector corresponds to the correct text chunk. Let's assume the extension order
            # from as_completed is acceptable for now, but acknowledge this potential issue.

            # Re-check lengths after processing
            if len(dense_vectors_list) != len(texts):
                logger.error(
                    f"Mismatch in number of dense vectors: expected {len(texts)}, got {len(dense_vectors_list)}"
                )
                # Handle error: maybe some tasks failed silently or logic error
                return  # or raise exception
            if len(sparse_vectors_list) != len(texts):
                logger.error(
                    f"Mismatch in number of sparse vectors: expected {len(texts)}, got {len(sparse_vectors_list)}"
                )
                # Handle error
                return  # or raise exception

            logger.info(
                f"Generated {len(dense_vectors_list)} dense and {len(sparse_vectors_list)} sparse vectors."
            )

            # Upload to Qdrant
            logger.info(f"Uploading {len(chunks)} chunks to Qdrant")
            points = []

            # Ensure sparse_vectors_list contains SparseEmbedding objects before accessing attributes
            if not all(isinstance(emb, SparseEmbedding) for emb in sparse_vectors_list):
                logger.error(
                    "Sparse vector list does not contain expected SparseEmbedding objects."
                )
                # Handle error appropriately, e.g., raise an exception or return
                raise TypeError("Unexpected type found in sparse_vectors_list")

            for i, chunk in enumerate(chunks):
                # Access indices and values from the SparseEmbedding object
                sparse_embedding_obj = sparse_vectors_list[i]
                # Add type check for safety, although the worker function type hint helps
                if not isinstance(sparse_embedding_obj, SparseEmbedding):
                    logger.error(
                        f"Item at index {i} is not a SparseEmbedding object: {type(sparse_embedding_obj)}"
                    )
                    continue  # Skip this point or handle error

                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vectors_list[i],
                            "sparse": models.SparseVector(
                                indices=sparse_embedding_obj.indices.tolist(),
                                values=sparse_embedding_obj.values.tolist(),
                            ),
                        },
                        payload={
                            "document_name": chunk["document_name"],
                            "section_title": chunk["section_title"],
                            "page_num": chunk["page_num"],
                            "page_end": chunk["page_end"],
                            "text": chunk["text"],
                            "source_file": pdf_path,
                            "ingestion_timestamp": str(datetime.now()),
                        },
                    )
                )

            # Upload in batches to avoid memory issues
            batch_size = 100
            logger.info(f"Upserting points in batches of {batch_size}...")
            for i in tqdm(
                range(0, len(points), batch_size), desc="Uploading to Qdrant"
            ):
                batch_points = points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch_points
                )

            logger.info(f"Successfully ingested {pdf_path}")

        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF into Qdrant vector database"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        logger.error(f"File not found: {args.pdf_path}")
        return

    if not args.pdf_path.lower().endswith(".pdf"):
        logger.error(f"Not a PDF file: {args.pdf_path}")
        return

    ingester = PDFIngester()
    ingester.ingest_pdf(args.pdf_path)

    logger.info("PDF ingestion complete")


if __name__ == "__main__":
    main()

# python ingestion.py /path/to/your/document.pdf
