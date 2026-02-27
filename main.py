import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import voyageai
import chromadb
from chromadb.config import Settings
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="FastAPI RAG Q&A")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "fastapi_docs"
VOYAGE_MODEL = "voyage-2"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K = 5

# Initialize clients
voyage_api_key = os.getenv("VOYAGE_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not voyage_api_key:
    raise ValueError("VOYAGE_API_KEY not found in .env file")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

vo_client = voyageai.Client(api_key=voyage_api_key)
claude_client = Anthropic(api_key=anthropic_api_key)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Get collection
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    raise RuntimeError(
        f"ChromaDB collection '{COLLECTION_NAME}' not found. "
        f"Please run ingest.py first. Error: {e}"
    )


# Request/Response models
class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    chunk_index: int


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "FastAPI RAG Q&A API",
        "collection_count": collection.count()
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a question using RAG with FastAPI documentation.

    1. Embeds the question using Voyage AI
    2. Retrieves relevant chunks from ChromaDB
    3. Sends to Claude with context for answer generation
    """
    try:
        # Step 1: Embed the question
        print(f"Embedding question: {request.question}")
        question_embedding = vo_client.embed(
            [request.question],
            model=VOYAGE_MODEL,
            input_type="query"
        ).embeddings[0]

        # Step 2: Query ChromaDB for relevant chunks
        print(f"Querying ChromaDB for top {TOP_K} relevant chunks...")
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K
        )

        # Extract retrieved chunks and metadata
        retrieved_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([
            f"Source: {meta['source']}\n{chunk}"
            for chunk, meta in zip(retrieved_chunks, metadatas)
        ])

        # Step 3: Build prompt for Claude
        system_prompt = """You are a helpful AI assistant that answers questions about FastAPI based on the official documentation.

Use the provided documentation excerpts to answer the user's question accurately and comprehensively.

If the documentation doesn't contain enough information to fully answer the question, say so clearly.

Keep your answers concise but complete."""

        user_prompt = f"""Based on the following FastAPI documentation excerpts, please answer the question.

Documentation excerpts:
{context}

Question: {request.question}

Answer:"""

        # Step 4: Call Claude API
        print("Calling Claude API...")
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.content[0].text

        # Prepare sources list
        sources = [
            Source(
                source=meta["source"],
                chunk_index=meta["chunk_index"]
            )
            for meta in metadatas
        ]

        print(f"✓ Answer generated successfully")

        return AskResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """
    Answer a question using RAG with streaming response.

    Returns a text/event-stream with Claude's response chunks as they arrive.
    """
    try:
        # Step 1: Embed the question
        print(f"Embedding question: {request.question}")
        question_embedding = vo_client.embed(
            [request.question],
            model=VOYAGE_MODEL,
            input_type="query"
        ).embeddings[0]

        # Step 2: Query ChromaDB for relevant chunks
        print(f"Querying ChromaDB for top {TOP_K} relevant chunks...")
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K
        )

        # Extract retrieved chunks and metadata
        retrieved_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([
            f"Source: {meta['source']}\n{chunk}"
            for chunk, meta in zip(retrieved_chunks, metadatas)
        ])

        # Step 3: Build prompt for Claude
        system_prompt = """You are a helpful AI assistant that answers questions about FastAPI based on the official documentation.

Use the provided documentation excerpts to answer the user's question accurately and comprehensively.

If the documentation doesn't contain enough information to fully answer the question, say so clearly.

Keep your answers concise but complete."""

        user_prompt = f"""Based on the following FastAPI documentation excerpts, please answer the question.

Documentation excerpts:
{context}

Question: {request.question}

Answer:"""

        # Step 4: Create streaming generator
        def generate_stream():
            print("Calling Claude API with streaming...")

            with claude_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    # Yield each chunk immediately as it arrives
                    yield text

            print(f"✓ Streaming completed")

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain"
        )

    except Exception as e:
        print(f"Error processing streaming question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
