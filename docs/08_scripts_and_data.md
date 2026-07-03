# 08: Scripts and Maintenance

## Model Development & Data
- **`data/extraction_train.jsonl`**: The primary training source containing generalized social media extraction examples.

## Ingestion & DB Management
- **`scripts/ingest.py`**: Populates ChromaDB with documents from local JSON/text sources.
- **`scripts/update_rag_telegram.py`**: Background service that fetches live messages from Telegram and adds them to the RAG context.
- **`scripts/inspect_db.py`**: CLI tool for debugging the vector database and checking ingested content.

## Environment & Health
- **`scripts/diagnose_torch.py`**: Ensures the GPU is correctly detected and has enough VRAM for the 4-bit Phi-3.5 model.
- **`scripts/fix_cuda.py`**: Utility for handling CUDA environment pathing on Windows systems.

## Configuration
- **`src/config.py`**: The central configuration file.
    - **`GGUF_MODEL_PATH`**: Points to the native model used by the main API.
    - **`LORA_ADAPTER_PATH`**: Points to the LoRA adapters used by specialized agents.
    - **`VECTOR_DB_DIR`**: Path to the ChromaDB storage.
- **`.env.example`**: Documentation for required secret keys like `GEMINI_API_KEY`.
