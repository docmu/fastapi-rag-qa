# FastAPI RAG Q&A System

This project is a Retrieval-Augmented Generation (RAG) AI question-answering system built with production-style backend architecture.

The application allows users to ask questions about documentation and receive AI-generated answers grounded in retrieved knowledge rather than model hallucinations.

The system combines semantic search with large language models using Anthropic Claude and embedding-based retrieval powered by Voyage AI.


https://github.com/user-attachments/assets/7c8d5f7c-fc89-417d-9af1-db0ad4f79f16



## What This Project Demonstrates 

- Building production-style AI backend APIs
- Designing Retrieval-Augmented Generation pipelines
- Working with vector databases and embeddings
- Implementing streaming AI responses
- Handling prompt engineering constraints
- Managing asynchronous AI workloads

## Tech Stack
- Backend Framework → FastAPI
- Vector Database → ChromaDB
- LLM Provider → Claude models via Anthropic API
- Embedding Model → Voyage AI semantic embeddings

## System Architecture

The system follows a standard RAG pipeline:
```
User Question
↓
Question Embedding Generation
↓
Semantic Search in Vector Database
↓
Top-K Knowledge Retrieval
↓
Prompt Construction with Context
↓
LLM Response Generation
↓
Streaming Output to Client
```
## Key Technical Features
### 1. Retrieval-Augmented Generation (RAG)

The system improves answer accuracy by combining:
- Semantic retrieval of documentation chunks
- Context-aware language generation
- Source attribution
Instead of relying solely on model training data, answers are grounded in retrieved knowledge.

### 2. Vector Search over Documentation
Documentation is processed through a pipeline:
- Document ingestion
- Text chunking with overlap for context preservation
- Embedding generation
- Storage in vector database
Semantic similarity is computed using cosine distance.

### 3. Streaming AI Responses
The API supports real-time token streaming using server-sent response chunks, improving perceived latency and user experience.

### 4. Production-Style Prompt Engineering
The system enforces response quality by:
- Providing explicit system-level instructions
- Restricting hallucinated knowledge
- Requiring answers to be based on retrieved documentation

## Ingestion Pipeline
The ingestion system:
- Clones documentation sources
- Splits text into ~500-token chunks with overlap
- Generates embeddings in batches
- Stores knowledge representations in vector storage
Chunk overlap is used to preserve semantic continuity across boundaries.
