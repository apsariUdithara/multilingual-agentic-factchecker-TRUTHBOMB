# 02: Backend API (FastAPI)

## File: `src/main.py`

This file serves as the primary entry point for the web services. It hosts the FastAPI application and manages the lifecycle of various AI models and database connections.

### Key Components

#### 1. System Initialization
- **Vector Database**: Initializes ChromaDB with `HuggingFaceEmbeddings` (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Linguistic Extractor**: Loads the `LinguisticExtractor` model for claim normalization.
- **BM25 Retriever**: A lexical search index built on top of the documents in ChromaDB for hybrid retrieval.

#### 2. Telegram Ingestion Worker
- **Function**: `telegram_ingestion_worker()`
- **Logic**: 
  - Uses `Telethon` to connect to Telegram.
  - Polls specified channels (e.g., `AdaDeranaNews`) every 60 seconds.
  - Deduplicates messages using `message_id` stored in ChromaDB metadata.
  - Injects new alerts as `Document` objects into the vector database.

#### 3. Fact-Checking Logic (LLM + GBNF)
- **Function**: `execute_fact_checker_llm(claim: str, context_documents: List[str])`
- **Syntax**: Uses `GBNF_GRAMMAR_STR` to enforce a strict JSON output format from the LLM.
- **Flow**:
  1. Combines the claim and retrieved context into a prompt.
  2. Executes the LLM with a temperature of 0.0 for deterministic results.
  3. Parses the output into a structured dictionary containing `verdict`, `confidence_score`, and `reasoning`.

### API Endpoints

#### `POST /api/v1/verify`
The main verification endpoint.
- **Input**: `VerificationRequest` (text and optional metadata hint).
- **Workflow**:
  1. **Extraction**: Calls `extractor_model.extract_claim()` to get a normalized English claim.
  2. **Hybrid Search**: Performs a similarity search in ChromaDB and a lexical search via BM25.
  3. **Analysis**: Passes the claim and context to the LLM fact-checker.
  4. **Response**: Returns a `VerificationResponse` with the verdict and retrieved sources.

#### `GET /api/v1/health`
Checks the status of the server and database connectivity.

### Important Global Variables
- `GBNF_GRAMMAR_STR`: A grammar definition that ensures the LLM returns valid JSON with specific keys (`verdict`, `confidence_score`, `reasoning`).
