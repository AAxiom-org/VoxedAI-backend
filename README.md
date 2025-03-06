# VoxAI Backend

A high-performance, modular Python FastAPI backend that efficiently handles LLM interactions, vector storage, and retrieval augmented generation (RAG). The system prioritizes startup-time processing over runtime computation, with optimized endpoints for minimal frontend latency.

## Project Overview

> **"Build a high-performance, modular Python FastAPI backend that efficiently handles LLM interactions, vector storage, and retrieval augmented generation (RAG). Prioritize startup-time processing over runtime computation, with optimized endpoints for minimal frontend latency."**

## Core Stack

> **"Backend: Python 3.11+, FastAPI; Database: Pinecone (vector DB), Supabase; Dependencies: pydantic for validation, httpx for async HTTP requests, asyncio for concurrent operations, uvicorn for ASGI server."**

## File Processors

The system includes a comprehensive set of file processors for extracting text and metadata from various file types:

- **Text Processor**: Handles plain text and markdown files
- **Document Processor**: Processes Word documents (DOCX)
- **PDF Processor**: Extracts text and metadata from PDF files
- **Spreadsheet Processor**: Handles Excel files (XLSX) and CSV files
- **Presentation Processor**: Processes PowerPoint files (PPTX)
- **Image Processor**: Extracts text and descriptions from images using Google Gemini
- **Audio Processor**: Transcribes audio files using Whisper
- **Video Processor**: Extracts audio transcription and frame descriptions from video files

## Setup Instructions

### Prerequisites

- Python 3.11+
- Pinecone API key
- Supabase credentials
- LLM API keys (Gemini, Anthropic, OpenAI)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -e .
   ```
3. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit `.env` to add your API keys for:
   - Pinecone
   - Supabase
   - OpenAI
   - Google Gemini

## Testing File Processors

To test the file processors:

1. Create a directory called `test_files` in the project root
2. Add sample files of different types to this directory
3. Run the test script:
   ```
   python app/tests/test_processors.py
   ```

This will process each file in the `test_files` directory with the appropriate processor and display the extracted content and metadata.

## API Design Principles

1. Minimize payload sizes. 2) Use async endpoints for I/O bound operations. 3) Implement caching strategies for repetitive requests. 4) Batch operations when possible. 5) Implement proper error handling with standardized response formats. 6) Use dependency injection for services. 7) Version all API endpoints. 8) Document APIs with OpenAPI/Swagger.

## Performance Optimization

1. Preload and cache embeddings at startup. 2) Use connection pooling for database access. 3) Implement background tasks for non-blocking operations. 4) Utilize async I/O for external calls. 5) Employ data streaming for large responses. 6) Minimize database round-trips by batching queries. 7) Implement proper indexing for vector search. 8) Use process-based parallelism for CPU-bound tasks.

## Database

1. Utilize Pinecone's in-house embedding models for vector generation and storage with optimal dimensionality and metric type. 2) Apply effective chunking strategies with context windows to enhance retrieval quality. 3) Leverage hybrid search by combining Pinecone's vector similarity and metadata filtering. 4) Organize data efficiently using namespaces for different document types. 5) Optimize ingestion with batched upsert operations. 6) Implement TTL for managing ephemeral vectors effectively. 7) Tune retrieval performance by adjusting top_k values for latency optimization. 8) Enhance search relevance with query preprocessing before embedding.

## Security Best Practices

1. Implement JWT-based authentication with Supabase. 2) Use CORS with appropriate origins. 3) Implement rate limiting. 4) Validate all input data with Pydantic. 5) Sanitize user inputs. 6) Implement proper error handling that doesn't leak sensitive information. 7) Use environmental variables for secrets. 8) Follow principle of least privilege for database access.

## Running the Application

Start the development server:

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation is available at http://localhost:8000/docs

## Development

See [Plan.md](Plan.md) for the detailed development plan and progress tracking.

## Performance Considerations

> **"1) Preload and cache embeddings at startup. 2) Use connection pooling for database access. 3) Implement background tasks for non-blocking operations. 4) Utilize async I/O for external calls. 5) Employ data streaming for large responses. 6) Minimize database round-trips by batching queries. 7) Implement proper indexing for vector search. 8) Use process-based parallelism for CPU-bound tasks."**
