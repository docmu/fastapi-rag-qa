# FastAPI RAG Q&A System

An educational project demonstrating **Retrieval-Augmented Generation (RAG)** - a fundamental AI development pattern that combines semantic search with large language models to answer questions using a custom knowledge base.

## What You'll Learn

This project teaches core AI development concepts:

1. **Vector Embeddings** - Converting text into numerical representations for semantic search
2. **Vector Databases** - Storing and querying embeddings efficiently
3. **RAG Pattern** - Combining retrieval with generation for accurate, context-aware responses
4. **Streaming Responses** - Real-time AI response delivery
5. **Production API Design** - Building scalable AI-powered APIs

## How RAG Works

### The Problem RAG Solves

Large Language Models (LLMs) like Claude have a knowledge cutoff and can't access your private data. RAG solves this by:

1. **Retrieval** - Finding relevant information from your knowledge base
2. **Augmentation** - Adding that information to the LLM's context
3. **Generation** - LLM generates an answer based on the provided context

### The RAG Pipeline

```
User Question
    ↓
[1] Convert question to vector embedding (Voyage AI)
    ↓
[2] Search vector database for similar embeddings (ChromaDB)
    ↓
[3] Retrieve top K most relevant text chunks
    ↓
[4] Build prompt: System instructions + Retrieved context + Question
    ↓
[5] Send to LLM for answer generation (Claude)
    ↓
[6] Stream response back to user
```

## Project Architecture

### Components

1. **`ingest.py`** - Data ingestion pipeline
   - Clones FastAPI documentation from GitHub
   - Chunks documents into digestible pieces
   - Generates embeddings for each chunk
   - Stores in vector database

2. **`main.py`** - FastAPI application
   - `/ask` - Standard Q&A endpoint (returns complete response)
   - `/ask/stream` - Streaming Q&A endpoint (real-time response)

3. **Vector Database (ChromaDB)** - Stores embeddings and enables semantic search

4. **Embedding Model (Voyage AI)** - Converts text to vectors

5. **LLM (Claude)** - Generates natural language answers

## Deep Dive: How Each Component Works

### 1. Text Chunking (`ingest.py`)

**Why chunk?**
- Embeddings work best on focused, coherent text segments
- LLMs have context limits (can't process entire documentation at once)
- Smaller chunks = more precise retrieval

**How it works:**
```python
# Example: 500 token chunks with 50 token overlap
Text: "AAAAAABBBBBBCCCCCCDDDDDD"
Chunk 1: "AAAAAABBBBBB"
Chunk 2:       "BBBBBBCCCCCC"  # Overlaps to preserve context
Chunk 3:              "CCCCCCDDDDDD"
```

**Key parameters:**
- `CHUNK_SIZE = 500` tokens (~2000 characters)
- `CHUNK_OVERLAP = 50` tokens (~200 characters)

**Why overlap is critical:**

Without overlap, information at chunk boundaries gets fragmented:

```
Document: "FastAPI uses Pydantic models for request validation. These models
           provide automatic data parsing and type checking."

Without overlap:
  Chunk 1: "FastAPI uses Pydantic models for request validation."
  Chunk 2: "These models provide automatic data parsing and type checking."
  ❌ "These models" loses reference to "Pydantic models"

With overlap (10 tokens):
  Chunk 1: "FastAPI uses Pydantic models for request validation. These models"
  Chunk 2: "request validation. These models provide automatic data parsing"
  ✓ Both chunks maintain full context about Pydantic models
```

**What breaks without overlap:**
- **Coreference resolution fails** - Pronouns (it, they, this) lose their referents
- **Incomplete thoughts** - Sentences split mid-concept
- **Missing connections** - Relationships between ideas severed
- **Lower retrieval accuracy** - Questions about "Pydantic validation" might miss Chunk 2

**The overlap trade-off:**
- Too little (0-5%): Context lost, poor retrieval
- Sweet spot (10%): Preserves context, minimal redundancy
- Too much (50%+): Wasteful storage, slower search

### 2. Vector Embeddings

**What are embeddings?**
Embeddings convert text into high-dimensional vectors (arrays of numbers) where semantically similar text has similar vectors.

```python
"How do I create a route?" → [0.2, -0.5, 0.8, ..., 0.3]  # 1024 dimensions
"Making a new endpoint"    → [0.3, -0.4, 0.7, ..., 0.4]  # Similar vector!
"What's the weather?"      → [-0.8, 0.2, -0.3, ..., 0.1] # Different vector
```

**Why use the same model for questions AND documents?**

This is critical - you MUST use the same embedding model for both:

```python
# During ingestion (ingest.py):
doc_embedding = voyage.embed("Creating routes with @app.get()",
                             input_type="document")
# → [0.2, 0.8, -0.3, ...]

# During query (main.py):
query_embedding = voyage.embed("How do I make an endpoint?",
                               input_type="query")
# → [0.25, 0.75, -0.28, ...]  # Similar vector!
```

**What breaks if you use different models:**

```python
# Ingestion with voyage-2:
doc = voyage_v2.embed("FastAPI routing")
# → [0.2, 0.8, -0.3, ...] in voyage-2's vector space

# Query with openai-ada:
query = openai.embed("How do I route?")
# → [12.5, -8.2, 3.1, ...] in OpenAI's vector space

similarity(doc, query) → 0.02 (no match!)
# ❌ Comparing vectors from different spaces is meaningless!
```

**Why this fails:**
- Each model learns its own **vector space** during training
- "Similar" in one space ≠ "similar" in another space
- Like comparing Celsius to Fahrenheit without conversion
- The dimensions don't align - they mean different things

**The `input_type` parameter:**

Voyage AI (and some other models) optimize embeddings differently:

- `input_type="document"` - Optimized for being searched
  - Encodes comprehensive meaning
  - Broader semantic representation

- `input_type="query"` - Optimized for searching
  - Encodes intent and specificity
  - Focused on what user wants to find

Both types live in the SAME vector space, but are optimized for their roles in retrieval.

**The magic of embeddings:**
```python
# Traditional keyword search fails here:
Query: "How do I make an endpoint?"
Doc: "Creating routes in FastAPI"  # No matching keywords!

# But embeddings understand meaning:
Query embedding:    [0.2, 0.8, -0.3, ...]
Doc embedding:      [0.3, 0.7, -0.2, ...]
Similarity score: 0.95 (very similar!) ✓
```

### 3. Vector Database (ChromaDB)

**What it does:**
Stores embeddings and performs **approximate nearest neighbor (ANN)** search to find similar vectors quickly.

**How semantic search works:**
```python
# 1. Store documents with embeddings
collection.add(
    ids=["chunk_0", "chunk_1", ...],
    embeddings=[[0.2, 0.8, ...], [0.1, 0.3, ...], ...],
    documents=["text content...", "more text...", ...],
    metadatas=[{"source": "intro.md"}, {"source": "advanced.md"}, ...]
)

# 2. Query with question embedding
results = collection.query(
    query_embeddings=[[0.25, 0.75, ...]],  # Your question as vector
    n_results=5  # Return top 5 most similar chunks
)
```

**Similarity measurement:**
ChromaDB uses cosine similarity to compare vectors:
```
similarity = (A · B) / (||A|| × ||B||)
```
- 1.0 = identical
- 0.0 = unrelated
- -1.0 = opposite

**What similarity search actually returns:**

When you query for the top K results, you get:

```python
results = collection.query(
    query_embeddings=[[0.25, 0.75, ...]],
    n_results=5
)

# Returns:
{
  'documents': [
    ["chunk about routing...", "chunk about dependencies...", ...],
  ],
  'metadatas': [
    [{"source": "routing.md"}, {"source": "deps.md"}, ...],
  ],
  'distances': [[0.12, 0.18, 0.24, 0.31, 0.42]]  # Lower = more similar
}
```

**Critical insight:** You get the top K by similarity score, NOT all matches above a threshold.

```python
# What you might expect:
"Return all chunks with similarity > 0.8"

# What actually happens:
"Return the 5 most similar chunks, even if they're not very similar"
```

**Failure modes of similarity search:**

1. **Out-of-domain queries:**
```python
Knowledge base: FastAPI documentation
Query: "How do I make chocolate chip cookies?"

# Still returns 5 chunks!
# They'll be about FastAPI, just the "most similar" ones
# Similarity scores will be low (0.3-0.5) but you still get results
```

**What breaks:** No "none of this is relevant" signal - LLM gets irrelevant context.

2. **Vocabulary mismatch:**
```python
Documentation uses: "path operation", "route handler"
User asks: "How do I make a REST endpoint?"

# Might miss relevant chunks if embeddings don't bridge the gap
# Embedding quality determines if "endpoint" → "path operation"
```

**What breaks:** Relevant info exists but isn't retrieved.

3. **Multi-hop reasoning:**
```python
Query: "How do I add authentication to my database queries?"

# Needs two separate concepts:
# - Authentication middleware
# - Database query setup

# Similarity search returns chunks about ONE or the OTHER
# Misses that you need to combine both concepts
```

**What breaks:** Each chunk addresses part of the question, full answer requires synthesis.

4. **Recency bias in embeddings:**
```python
Recent doc: "Use the new @app.post() decorator"
Old doc: "The old @app.route() method is deprecated"

Query: "How do I create endpoints?"

# Both chunks are similar!
# No inherent way to prefer newer information
```

**What breaks:** Outdated info returned with equal weight.

5. **The "K" problem:**
```python
K=1:  Might miss important info (only one perspective)
K=5:  Good balance (the default in this project)
K=20: Noise overwhelms signal, LLM context window wasted
```

**Trade-off:** More chunks = more context but also more noise and token cost.

### 4. The RAG Prompt Pattern

**Building effective context:**

```python
system_prompt = """
You are an AI assistant that answers questions about FastAPI.
Use ONLY the provided documentation excerpts.
If the docs don't have the info, say so clearly.
"""

user_prompt = f"""
Documentation excerpts:
---
Source: tutorial/first-steps.md
Create routes with @app.get("/") decorator...
---
Source: advanced/dependencies.md
FastAPI's dependency injection system...
---

Question: {user_question}

Answer:
"""
```

**Why this prompt structure works:**

The ordering is deliberate and critical:

```python
1. System prompt (sets the rules)
2. Retrieved documentation (the facts)
3. User question (what to answer)
4. "Answer:" (triggers response)
```

**Why ordering matters:**

**Primacy effect** - LLMs pay more attention to information early in the context:

```python
# GOOD: Facts before question
"""
Context: FastAPI uses @app.get() for GET requests.
Question: How do I handle GET requests?
"""
# → LLM sees facts first, uses them to answer

# BAD: Question before facts
"""
Question: How do I handle GET requests?
Context: FastAPI uses @app.get() for GET requests.
"""
# → LLM might start answering before seeing the facts
```

**Recency effect** - LLMs also emphasize recent information:

```python
# Our structure:
Documentation → Question → "Answer:"
                    ↑           ↑
              (guides focus) (triggers)

# The question being recent reminds the LLM what to answer
# "Answer:" immediately triggers generation
```

**Why "Use ONLY the provided documentation":**

Without this constraint:

```python
# Without constraint:
User: "How do I deploy FastAPI?"
# → LLM might hallucinate: "Use Heroku with git push..."
# (Even if docs don't mention deployment)

# With constraint:
User: "How do I deploy FastAPI?"
# → LLM: "The provided documentation doesn't cover deployment."
# ✓ Honest, verifiable answer
```

**What breaks without proper structure:**

1. **No system prompt:**
   - LLM might ignore retrieved docs and use training data
   - Can't control behavior or set constraints
   - Inconsistent output format

2. **Context after question:**
   - LLM starts answering before seeing all facts
   - Might miss crucial information
   - Lower accuracy

3. **No source citations:**
   - Can't verify answers
   - Can't trace errors back to source
   - User can't learn more

4. **Retrieved chunks in random order:**
   ```python
   # Unordered (by similarity desc):
   Chunk 1: Score 0.95 - "FastAPI uses Pydantic for validation"
   Chunk 2: Score 0.87 - "Validation happens automatically"
   Chunk 3: Score 0.82 - "Pydantic models define schemas"

   # Ordered by relevance gives LLM the best info first
   # Most relevant facts are freshest in model's attention
   ```

**The complete pattern:**

```python
system_prompt = """
[1. Define role and constraints]
You are a helpful assistant.
Use ONLY the provided docs.
If info is missing, say so.
"""

user_prompt = f"""
[2. Inject retrieved context with sources]
Documentation:
---
Source: {source1}
{most_relevant_chunk}
---
Source: {source2}
{second_most_relevant}
---

[3. State the question]
Question: {user_question}

[4. Trigger generation]
Answer:
"""
```

This structure:
- ✓ Establishes constraints first
- ✓ Provides facts before asking
- ✓ Enables verification via sources
- ✓ Puts most relevant info early (primacy)
- ✓ Keeps question recent (recency)
- ✓ Cleanly separates concerns

### 5. Streaming Responses

**Why UX demands streaming:**

Consider the non-streaming experience:

```python
# Non-streaming:
User asks question → [5 seconds of nothing] → Full answer appears

# User experience:
- "Is it working?"
- "Did my request fail?"
- "Should I refresh?"
- Feels broken, slow, unresponsive
```

With streaming:

```python
# Streaming:
User asks question → [0.5s] → "To create" → "a FastAPI" → "route, use"...

# User experience:
- Immediate feedback
- Progress visible
- Can start reading while generation continues
- Feels fast, responsive, modern
```

**The psychology:**

```
Actual generation time: 5 seconds (both approaches)

Perceived wait time:
- Non-streaming: 5 seconds of anxiety
- Streaming: 0.5 seconds until first token (then engaging content)

Perceived speed: ~10x better with streaming
```

**Technical: How streaming works:**

**Server-Sent Events (SSE) / Chunked Transfer:**

```python
# Traditional HTTP:
Client ← [complete response] ← Server
        (waits for all bytes)

# Streaming HTTP:
Client ← "chunk1" ← Server  # Immediately
Client ← "chunk2" ← Server  # As generated
Client ← "chunk3" ← Server  # Real-time
Client ← "done"   ← Server
```

**Implementation levels:**

1. **LLM API level** (Anthropic):
```python
with claude_client.messages.stream(...) as stream:
    for text in stream.text_stream:
        # Claude sends tokens as they're generated
        # text = "To", then "create", then "a"...
        yield text
```

2. **FastAPI level**:
```python
def generate_stream():
    for chunk in llm_stream:
        yield chunk  # Python generator

return StreamingResponse(generate_stream())
# FastAPI sends each yield immediately to client
```

3. **Frontend level**:
```javascript
const reader = response.body.getReader()
while (true) {
  const {value, done} = await reader.read()
  if (done) break
  // Display chunk immediately
  displayChunk(decode(value))
}
```

**What breaks without streaming:**

1. **Timeout errors:**
```python
# Long response (10 seconds)
# Many proxies timeout at 30s, some at 10s
# Non-streaming holds connection open
# → "504 Gateway Timeout" before response completes
```

2. **Memory issues:**
```python
# Non-streaming:
full_response = generate_complete_answer()  # 50KB in memory
return full_response

# Streaming:
for chunk in generate_answer():
    yield chunk  # Only ~100 bytes in memory at once
    # Previous chunks already sent and freed
```

3. **Poor mobile experience:**
```python
# Mobile connections are slow
# Non-streaming: User sees nothing for 8+ seconds on 3G
# Streaming: User sees first token in 1-2 seconds
```

4. **Can't cancel generation:**
```python
# Non-streaming: Once you make the request, you wait
# No way to stop if answer is going wrong direction

# Streaming: User can cancel mid-generation
# Saves compute and cost
```

**The streaming pattern:**

```python
# Server (FastAPI):
def generate_stream():
    # Generator function - yields one chunk at a time
    with claude_client.messages.stream(...) as stream:
        for text in stream.text_stream:
            yield text  # Each yield sends immediately

return StreamingResponse(
    generate_stream(),
    media_type="text/plain"  # Plain text chunks
)

# Client (Browser):
const response = await fetch('/ask/stream', {
    method: 'POST',
    body: JSON.stringify({question})
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
    const {done, value} = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    displayText(chunk)  // Show immediately
}
```

**Why this is the standard now:**

Every modern AI product streams:
- ChatGPT
- Claude
- Cursor
- GitHub Copilot

Because:
- ✓ 10x better perceived performance
- ✓ User knows it's working
- ✓ Can start reading sooner
- ✓ Can cancel if needed
- ✓ Better error handling
- ✓ Lower memory footprint

## Key AI Development Concepts

### 1. Semantic Search vs Keyword Search

**Keyword search:**
```
Query: "create endpoint"
Matches: Documents containing exact words "create" AND "endpoint"
Misses: "Building routes", "Define API paths"
```

**Semantic search:**
```
Query: "create endpoint"
Matches: Documents about routes, endpoints, decorators, path operations
(Understands meaning, not just keywords)
```

### 2. The Importance of Chunking Strategy

**Too small (100 tokens):**
- ❌ Loses context
- ❌ More chunks = slower retrieval
- ✓ Very precise matches

**Too large (2000 tokens):**
- ❌ Less precise retrieval
- ❌ May exceed LLM context limits
- ✓ More context preserved

**Sweet spot (500 tokens):**
- ✓ Balanced precision and context
- ✓ Fast retrieval
- ✓ Fits well in prompts

### 3. Retrieval Accuracy

**Factors affecting accuracy:**

1. **Number of chunks retrieved (`TOP_K`)**
   - Too few (K=1): Might miss important info
   - Too many (K=20): Adds noise, uses context budget
   - Sweet spot (K=5): Usually sufficient

2. **Embedding model quality**
   - Better models = better semantic understanding
   - Domain-specific models can help (e.g., code-specific embeddings)

3. **Chunk overlap**
   - Prevents information loss at boundaries
   - 10% overlap is typical (50/500 tokens)

### 4. Prompt Engineering for RAG

**Best practices:**

1. **Clear instructions:**
   ```python
   "Use ONLY the provided documentation"  # Prevents hallucination
   ```

2. **Source attribution:**
   ```python
   f"Source: {metadata['source']}\n{chunk}"  # Enables verification
   ```

3. **Explicit constraints:**
   ```python
   "If the docs don't contain the info, say so clearly"
   ```

4. **Structured format:**
   ```
   Documentation excerpts:
   ---
   [chunk 1]
   ---
   [chunk 2]

   Question: [question]
   ```

## Running the Project

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys in `.env`:**
   ```bash
   ANTHROPIC_API_KEY=your_claude_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   ```

### Ingest Documentation

```bash
python3 ingest.py
```

**What happens:**
1. Clones FastAPI docs (sparse checkout - only docs folder)
2. Reads all markdown files
3. Chunks into ~500 token pieces
4. Generates embeddings (with rate limiting for free tier)
5. Stores in ChromaDB at `./chroma_db/`

**Note:** With free tier Voyage AI limits (3 RPM, 10K TPM), this takes 1-4 hours. Add payment method for faster processing (still free - 200M tokens included).

### Run the API

```bash
python3 main.py
```

Server runs at `http://localhost:8000`

**Available endpoints:**

- `GET /` - Health check, shows collection size
- `POST /ask` - Standard Q&A (waits for complete response)
- `POST /ask/stream` - Streaming Q&A (real-time chunks)

### Test the System

**Using Swagger UI:**
1. Visit `http://localhost:8000/docs`
2. Try `/ask` endpoint
3. Enter: `{"question": "How do I create a FastAPI route?"}`

**Using curl:**
```bash
# Standard response
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a FastAPI route?"}'

# Streaming response
curl -N -X POST "http://localhost:8000/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create a FastAPI route?"}'
```

**Using Python:**
```bash
python3 test_stream.py
```

**Using the React Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173` to use the web interface with real-time streaming.

See [frontend/README.md](frontend/README.md) for more details.

## Understanding the Code

### Ingestion Pipeline (`ingest.py`)

```python
# 1. Clone docs
clone_fastapi_docs()
# Uses git sparse-checkout to get only docs/

# 2. Read markdown
documents = read_markdown_files()
# Recursively finds .md files

# 3. Chunk
chunks = chunk_documents(documents)
# Splits into 500-token pieces with 50-token overlap

# 4. Generate embeddings
embeddings = generate_embeddings(chunk_texts, api_key)
# Voyage AI: converts text → vectors
# Rate limited: 4 chunks per batch, 30s delay

# 5. Store in vector DB
store_in_chromadb(chunks, embeddings)
# ChromaDB: persists to ./chroma_db/
```

### Query Pipeline (`main.py`)

```python
# 1. Embed question
question_embedding = vo_client.embed(
    [request.question],
    model="voyage-2",
    input_type="query"  # Optimized for search
).embeddings[0]

# 2. Semantic search
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5  # Top 5 most similar chunks
)

# 3. Build context
context = "\n\n---\n\n".join([
    f"Source: {meta['source']}\n{chunk}"
    for chunk, meta in zip(retrieved_chunks, metadatas)
])

# 4. Create prompt
user_prompt = f"""
Documentation excerpts:
{context}

Question: {request.question}

Answer:"""

# 5. Get LLM response
# Non-streaming:
response = claude_client.messages.create(...)

# Streaming:
with claude_client.messages.stream(...) as stream:
    for text in stream.text_stream:
        yield text
```

## Common Patterns & Best Practices

### 1. Error Handling

```python
try:
    # Voyage AI can rate limit
    embeddings = vo_client.embed(batch)
except Exception as e:
    print(f"Error: {e}")
    time.sleep(60)  # Wait before retry
    embeddings = vo_client.embed(batch)  # Retry once
```

### 2. Rate Limiting

```python
# Free tier: 3 requests/min, 10K tokens/min
batch_size = 4  # ~2000 tokens (well under 10K)
delay_between_batches = 30  # 2 requests/min (buffer from 3)

for i in range(0, len(texts), batch_size):
    process_batch(texts[i:i+batch_size])
    time.sleep(delay_between_batches)
```

### 3. Metadata Tracking

```python
# Store source information with each chunk
metadata = {
    "source": "tutorial/first-steps.md",
    "chunk_index": 0,
    "total_chunks": 5
}

# Enables:
# - Source attribution in responses
# - Debugging retrieval issues
# - User verification of information
```

## Extending This Project

### Ideas for Learning More

1. **Experiment with chunking:**
   - Try different chunk sizes (250, 1000 tokens)
   - Vary overlap (0, 25, 100 tokens)
   - Compare retrieval accuracy

2. **Improve retrieval:**
   - Implement hybrid search (keyword + semantic)
   - Try re-ranking retrieved results
   - Experiment with different embedding models

3. **Add features:**
   - Conversation history (multi-turn chat)
   - Citation links in responses
   - Feedback mechanism (thumbs up/down)

4. **Optimize performance:**
   - Cache embeddings for common queries
   - Batch processing for multiple questions
   - Add Redis for session management

5. **Evaluate quality:**
   - Build a test set of Q&A pairs
   - Measure retrieval accuracy (precision/recall)
   - Track response quality metrics

## Key Takeaways

### What Makes RAG Powerful

1. **Accuracy** - LLM answers based on actual documentation, not memorized training data
2. **Up-to-date** - Update docs, re-run ingestion, instant knowledge update
3. **Transparency** - Returns sources, users can verify
4. **Privacy** - Your data stays in your vector DB, not sent for training

### When to Use RAG

✅ **Good for:**
- Q&A over documentation
- Customer support (company knowledge base)
- Research assistants (academic papers)
- Code search (internal codebases)

❌ **Not ideal for:**
- Creative writing (no retrieval needed)
- General knowledge questions (LLM alone is fine)
- Real-time data (need different approach)

### The RAG vs Fine-tuning Decision

**RAG:**
- ✓ Easy to update (just re-ingest)
- ✓ Transparent (shows sources)
- ✓ Works with any LLM
- ✓ Cheaper (no training costs)
- ❌ Requires good chunking/retrieval

**Fine-tuning:**
- ✓ Knowledge baked into model
- ✓ Can learn style/tone
- ❌ Expensive to update (retrain)
- ❌ Black box (no sources)
- ❌ Requires lots of data

**Often best:** RAG + fine-tuned model (RAG for facts, fine-tuning for style)

## Resources

### APIs Used
- [Anthropic Claude API](https://docs.anthropic.com/) - LLM for generation
- [Voyage AI](https://docs.voyageai.com/) - Text embeddings
- [ChromaDB](https://docs.trychroma.com/) - Vector database

### Further Reading
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Pinecone's guide
- [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - Understanding embeddings
- [Prompt Engineering](https://www.anthropic.com/index/prompting-best-practices) - Anthropic's guide

## License

Educational project - feel free to use and modify for learning purposes.
