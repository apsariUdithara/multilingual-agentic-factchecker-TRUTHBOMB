# 06: Tools and RAG Infrastructure

The system employs **Retrieval-Augmented Generation (RAG)** to ensure all verdicts are grounded in verifiable and high-credibility evidence.

## 1. Trusted Web Search Tool (`src/tools/web_search_tool.py`)
To prevent the collection of misinformation, the web search functionality is strictly restricted to authoritative domains.
- **Mechanism**: The tool automatically appends a domain whitelist to every search query using the `site:` operator.
- **Trusted Domains**:
  - `adaderana.lk`, `newsfirst.lk`, `dailymirror.lk`, `ft.lk`
  - `news.lk` (Official Government News Portal)
  - `lankadeepa.lk`, `virakesari.lk`
  - `economynext.com`, `island.lk`
- **Fallback**: If a restricted search fails, the system may perform a general search but flags the results as 'UNRESTRICTED', prompting the **Web Researcher Agent** to exercise extreme caution.

## 2. Real-Time OSINT Ingestion (Telegram)
The system continuously monitors official communication channels to populate its local database.
- **Monitored Channels**:
  - **News**: `@AdaDeranaNews`, `@newsfirstsl`, `@HiruNews`
  - **Official**: `@PMDNews` (President's Media Division), `@DGINews` (Dept. of Govt Information), `@news_lk`.
- **Deduplication**: Uses `message_id` metadata to ensure the same alert isn't processed twice.

## 3. Local Database Tool (`src/tools/local_db_tool.py`)
Searches through the internal repository of news and government data already ingested.
- **Search Mechanism**:
  - **Similarity Search**: Uses ChromaDB and multilingual embeddings to find semantically relevant alerts.
  - **Lexical Search**: BM25 keyword matching for specific entities (e.g., "PTA", "VAT", "VAT increase").
- **Reasoning Engine**: Uses a fine-tuned **Phi-3.5-mini** model (via LoRA adapters specified in `LORA_ADAPTER_PATH`) to reconcile claims against the specific snippets found in the local store.

## 4. Internal Knowledge Base (ChromaDB)
- **Data Persistence**: Stored in `pipeline-setup/vector_db`.
- **Multilingual Support**: Uses `paraphrase-multilingual-MiniLM-L12-v2` to allow claims in Sinhala/Singlish to match against English news reports.
