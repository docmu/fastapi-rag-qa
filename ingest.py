"""
FastAPI Documentation Ingestion Pipeline

This script implements the data preparation phase of a RAG (Retrieval-Augmented Generation) system.

What it does:
1. Clones FastAPI documentation from GitHub
2. Reads all markdown files
3. Chunks documents into ~500 token pieces with overlap
4. Generates vector embeddings for each chunk
5. Stores chunks + embeddings in ChromaDB vector database

Why each step matters:
- Cloning: Gets the source knowledge base (could be any documentation)
- Chunking: Makes text searchable at the right granularity
- Overlap: Prevents context loss at chunk boundaries
- Embeddings: Enables semantic search (meaning-based, not keyword)
- Storage: Makes everything queryable at runtime
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import voyageai  # Embedding model API
import chromadb  # Vector database
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Constants
REPO_URL = "https://github.com/fastapi/fastapi.git"
CLONE_DIR = "fastapi_repo"
DOCS_PATH = "docs/en/docs"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "fastapi_docs"

# Chunking parameters - critical for RAG performance
# CHUNK_SIZE: How much text per chunk
#   - Too small (100): Fragments context, too many chunks to search
#   - Too large (2000): Less precise retrieval, may exceed LLM limits
#   - Sweet spot (500): Balance of context and precision
CHUNK_SIZE = 500  # tokens

# CHUNK_OVERLAP: How much chunks share with neighbors
#   - Why needed: Prevents information loss at boundaries
#   - Example: "FastAPI uses Pydantic. Pydantic provides..." would split badly without overlap
#   - 10% overlap (50/500) is typical - preserves context without excessive redundancy
CHUNK_OVERLAP = 50  # tokens

# Embedding model - MUST be the same model used at query time
VOYAGE_MODEL = "voyage-2"

# Token estimation - embeddings work on tokens, not characters
# ~4 characters per token is a rough average for English text
# Used to convert CHUNK_SIZE (tokens) to character count for splitting
CHARS_PER_TOKEN = 4


def clone_fastapi_docs():
    """
    Clone docs/ folder only using git sparse checkout (5MB vs 50MB full repo).
    What breaks: Without this, no source data to process.
    """
    print(f"Cloning FastAPI docs from {REPO_URL}...")

    if os.path.exists(CLONE_DIR):
        shutil.rmtree(CLONE_DIR)

    os.makedirs(CLONE_DIR)
    subprocess.run(["git", "init"], cwd=CLONE_DIR, check=True)
    subprocess.run(["git", "remote", "add", "origin", REPO_URL], cwd=CLONE_DIR, check=True)

    # Sparse checkout: only pull docs/, not entire repo
    subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=CLONE_DIR, check=True)

    sparse_checkout_file = Path(CLONE_DIR) / ".git" / "info" / "sparse-checkout"
    sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_checkout_file.write_text("docs/\n")

    subprocess.run(["git", "pull", "origin", "master"], cwd=CLONE_DIR, check=True)
    print("✓ FastAPI docs cloned successfully")


def read_markdown_files() -> List[Dict[str, str]]:
    """
    Load all .md files recursively from docs directory.
    What breaks: Without path metadata, can't attribute sources in answers.
    """
    docs_dir = Path(CLONE_DIR) / DOCS_PATH
    markdown_files = []

    if not docs_dir.exists():
        print(f"Error: Directory {docs_dir} does not exist")
        return []

    print(f"Reading markdown files from {docs_dir}...")

    for md_file in docs_dir.rglob("*.md"):  # rglob = recursive glob
        try:
            content = md_file.read_text(encoding="utf-8")
            relative_path = md_file.relative_to(docs_dir)
            markdown_files.append({
                "path": str(relative_path),  # Stored for source attribution
                "content": content,
                "full_path": str(md_file)
            })
        except Exception as e:
            print(f"Error reading {md_file}: {e}")

    print(f"✓ Read {len(markdown_files)} markdown files")
    return markdown_files


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks using sliding window.

    Why overlap matters:
    Without: "FastAPI uses Pydantic. Pydantic validates..." → chunks lose connection
    With:    Both chunks contain "Pydantic", preserving context

    What breaks: Remove overlap → pronouns/references lose referents, worse retrieval
    """
    chunk_chars = chunk_size * CHARS_PER_TOKEN  # Convert tokens to chars
    overlap_chars = overlap * CHARS_PER_TOKEN

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]

        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)

        # Slide window forward by (chunk_size - overlap)
        # This creates overlap_chars of shared content between adjacent chunks
        start = end - overlap_chars

        if end >= len(text):
            break

    return chunks


def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Chunk all docs and attach metadata for source attribution.
    What breaks: Without chunk_index, can't identify which part of doc was used.
    """
    print(f"Chunking documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    chunked_docs = []

    for doc in documents:
        chunks = chunk_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "content": chunk,
                "metadata": {
                    "source": doc["path"],        # Which file
                    "chunk_index": i,              # Which part of file
                    "total_chunks": len(chunks)    # How many parts total
                }
            })

    print(f"✓ Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs


def generate_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Convert text to vectors using Voyage AI.

    CRITICAL: input_type="document" (different from "query" at search time)
    What breaks: Different model → vectors incomparable, no rate limiting → API errors
    """
    print(f"Generating embeddings using {VOYAGE_MODEL}...")
    print("Note: Using conservative rate limiting for free tier (3 RPM, 10K TPM)")

    vo_client = voyageai.Client(api_key=api_key)

    # Free tier: 3 req/min, 10K tokens/min
    # 4 chunks * 500 tokens = 2K tokens/req (safe under limit)
    batch_size = 4
    all_embeddings = []

    delay_between_batches = 30  # 30s = 2 req/min (buffer)

    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            # input_type="document" = optimize for being retrieved
            # MUST use same model at query time for vector space alignment
            result = vo_client.embed(
                batch,
                model=VOYAGE_MODEL,
                input_type="document"  # Not "query"
            )
            all_embeddings.extend(result.embeddings)
            print(f"  ✓ Batch {batch_num} complete ({len(all_embeddings)}/{len(texts)} total embeddings)")

            if i + batch_size < len(texts):
                print(f"  Waiting {delay_between_batches}s for rate limiting...")
                time.sleep(delay_between_batches)

        except Exception as e:
            print(f"  Error processing batch {batch_num}: {e}")
            print("  Waiting 60s before retrying...")
            time.sleep(60)

            try:
                result = vo_client.embed(batch, model=VOYAGE_MODEL, input_type="document")
                all_embeddings.extend(result.embeddings)
                print(f"  ✓ Batch {batch_num} complete on retry")
            except Exception as retry_error:
                print(f"  Failed again: {retry_error}")
                raise

    print(f"✓ Generated {len(all_embeddings)} embeddings")
    return all_embeddings


def store_in_chromadb(chunks: List[Dict[str, str]], embeddings: List[List[float]]):
    """
    Store chunks + embeddings in vector database for similarity search.
    What breaks: Misaligned chunks/embeddings → wrong results, no persistence → recompute every time
    """
    print(f"Storing data in ChromaDB at {CHROMA_DB_DIR}...")

    # PersistentClient = saves to disk (vs Client = in-memory only)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # Fresh start each run (delete old data)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass

    # Create new collection for this ingestion
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "FastAPI documentation chunks"}
    )

    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Add to collection in batches
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        print(f"Storing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")

        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")

    # Print collection info
    print(f"\nCollection '{COLLECTION_NAME}' created with {collection.count()} documents")


def main():
    """Main ingestion pipeline."""
    print("=== FastAPI Documentation Ingestion Pipeline ===\n")

    # Check for API key
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise ValueError("VOYAGE_API_KEY not found in .env file")

    # Step 1: Clone FastAPI docs
    clone_fastapi_docs()

    # Step 2: Read markdown files
    documents = read_markdown_files()
    if not documents:
        print("No documents found. Exiting.")
        return

    # Step 3: Chunk documents
    chunks = chunk_documents(documents)

    # Step 4: Generate embeddings
    chunk_texts = [chunk["content"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts, voyage_api_key)

    # Step 5: Store in ChromaDB
    store_in_chromadb(chunks, embeddings)

    print("\n✓ Ingestion pipeline completed successfully!")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Database location: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
