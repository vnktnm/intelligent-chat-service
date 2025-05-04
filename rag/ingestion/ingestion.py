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
from openai import OpenAI
from sentence_transformers import SentenceTransformer
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


class PDFIngester:
    def __init__(self):
        # Set up OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Set up Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )

        # Initialize sparse vectorizer model
        self.sparse_model = SentenceTransformer("naver/splade-cocondenser-selfdistil")

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

            # Create collection with hybrid search capability
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=3072,  # Embedding size for OpenAI text-embedding-3-large
                        distance=models.Distance.COSINE,
                    ),
                    "sparse": models.VectorParams(
                        size=30522,  # SPLADE model vocabulary size
                        distance=models.Distance.Dot,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                },
            )

            # Create payload index for efficient filtering
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

    def create_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create dense embeddings using OpenAI's embedding model"""
        logger.info("Creating dense embeddings")

        embeddings = []
        batch_size = 20  # Adjust based on API limits

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            response = self.openai_client.embeddings.create(
                input=batch_texts, model="text-embedding-3-large"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def create_dense_batch(self, batch_texts: List[str]) -> List[List[float]]:
        """Process a batch of texts for dense embeddings"""
        response = self.openai_client.embeddings.create(
            input=batch_texts, model="text-embedding-3-large"
        )
        return [item.embedding for item in response.data]

    def create_sparse_batch(self, batch_texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts for sparse embeddings"""
        embeddings = self.sparse_model.encode(batch_texts, convert_to_numpy=True)

        results = []
        for embedding in embeddings:
            # Get indices and values of non-zero elements
            indices = np.nonzero(embedding)[0].tolist()
            values = embedding[indices].tolist()

            # Create sparse vector representation
            results.append({"indices": indices, "values": values})

        return results

    def create_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Create sparse embeddings for titles/text"""
        logger.info("Creating sparse embeddings")

        sparse_embeddings = []
        batch_size = 32  # Adjust based on your GPU/CPU capabilities

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Get embeddings from the model
            embeddings = self.sparse_model.encode(batch_texts, convert_to_numpy=True)

            for embedding in embeddings:
                # Get indices and values of non-zero elements
                indices = np.nonzero(embedding)[0].tolist()
                values = embedding[indices].tolist()

                # Create sparse vector
                sparse_embeddings.append({"indices": indices, "values": values})

        return sparse_embeddings

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
            dense_batch_size = 20  # Adjust based on API limits
            sparse_batch_size = 32  # Adjust based on CPU/memory constraints

            dense_batches = [
                texts[i : i + dense_batch_size]
                for i in range(0, len(texts), dense_batch_size)
            ]
            sparse_batches = [
                texts[i : i + sparse_batch_size]
                for i in range(0, len(texts), sparse_batch_size)
            ]

            # Run dense and sparse embedding generation in parallel
            dense_vectors = []
            sparse_vectors = []

            # Using ThreadPoolExecutor for I/O bound API calls (OpenAI)
            # Using ProcessPoolExecutor for CPU-intensive operations (sparse vectors)
            with (
                concurrent.futures.ThreadPoolExecutor() as dense_executor,
                concurrent.futures.ProcessPoolExecutor() as sparse_executor,
            ):

                # Submit all dense batch tasks
                dense_futures = [
                    dense_executor.submit(self.create_dense_batch, batch)
                    for batch in dense_batches
                ]

                # Submit all sparse batch tasks
                sparse_futures = [
                    sparse_executor.submit(self.create_sparse_batch, batch)
                    for batch in sparse_batches
                ]

                # Collect dense results as they complete
                for future in concurrent.futures.as_completed(dense_futures):
                    try:
                        result = future.result()
                        dense_vectors.extend(result)
                    except Exception as e:
                        logger.error(f"Error in dense vector generation: {str(e)}")
                        raise

                # Collect sparse results as they complete
                for future in concurrent.futures.as_completed(sparse_futures):
                    try:
                        result = future.result()
                        sparse_vectors.extend(result)
                    except Exception as e:
                        logger.error(f"Error in sparse vector generation: {str(e)}")
                        raise

            # Ensure results are in the correct order (same as texts)
            assert len(dense_vectors) == len(
                texts
            ), "Mismatch in number of dense vectors"
            assert len(sparse_vectors) == len(
                texts
            ), "Mismatch in number of sparse vectors"

            # Upload to Qdrant
            logger.info(f"Uploading {len(chunks)} chunks to Qdrant")
            points = []

            for i, chunk in enumerate(chunks):
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vectors[i],
                            "sparse": models.SparseVector(
                                indices=sparse_vectors[i]["indices"],
                                values=sparse_vectors[i]["values"],
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
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch_points
                )

            logger.info(f"Successfully ingested {pdf_path}")

        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}")
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
