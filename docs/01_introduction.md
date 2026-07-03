# Documentation Series: Social Media Claim Fact-Checking Engine

## 01: Project Overview

### Introduction
The **Social Media Claim Fact-Checking Engine** is a generalized automated system designed to extract and verify claims from mixed linguistic streams (primarily Sri Lankan social media posts). While it previously specialized in traffic laws, it has been evolved into a robust platform for verifying news, political statements, and social issues across English, Sinhala, Tamil, and Singlish.

### Core Objectives
- **Linguistic Extraction**: Normalize complex, code-mixed, or emotional social media posts into professional English declarative claims using a custom fine-tuned model.
- **Hybrid Retrieval**: Search for evidence using a combination of semantic (vector-based) search and lexical (keyword-based) search across a diverse OSINT database.
- **Automated Verification**: Leverage Large Language Models (LLMs) to analyze claims against retrieved context and provide a structured verdict (TRUE, FALSE, or UNVERIFIED).
- **Multi-Agent Orchestration**: Use specialized agents to triage, research, and synthesize information for higher accuracy.

### System Architecture
1.  **Ingestion Layer**: Background workers (MTProto/Telethon) polling Telegram channels for real-time news alerts and social media streams.
2.  **Intelligence Layer (Model fine-tuned on Unsloth)**:
    -   A **Phi-3.5-mini-instruct** base model fine-tuned using the **Unsloth** library on a custom dataset (`extraction_train.jsonl`).
    -   Deployment via **llama.cpp** using GGUF quantization for efficient local inference.
3.  **Processing Layer**:
    -   **FastAPI Backend**: Provides a standard REST API for claim verification.
    -   **LangGraph Pipeline**: Orchestrates a multi-agent workflow to handle complex research tasks.
4.  **Storage Layer**: ChromaDB for vector embeddings and BM25 for keyword retrieval.
5.  **User Interface**:
    -   **Streamlit Dashboard**: Rapid testing and visualization interface.
    -   **React Web App**: Modern, component-based frontend for end-users.

### Tech Stack
- **Language**: Python 3.10+
- **Frameworks**: FastAPI, LangGraph, Streamlit
- **Fine-Tuning**: Unsloth, Google Colab (Tesla T4/A100)
- **Inference**: llama-cpp-python (GGUF Q4_K_M)
- **Database**: ChromaDB
- **Frontend**: React, TypeScript, Tailwind CSS
