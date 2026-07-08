# Cyber-Physical Social Media Fact-Checking Engine Documentation

Welcome to the documentation series for the Social Media Claim Fact-Checking Engine. This series provides a step-by-step technical explanation of the codebase, covering the generalized extraction model and the multi-agent orchestration.

## Documentation Modules

1.  **[01: Project Overview](01_introduction.md)**
    *   Generalized goals, system architecture, and tech stack.
2.  **[02: Backend API (FastAPI)](02_backend_fastapi.md)**
    *   Explanation of `src/main.py`, the core API endpoints, and the GBNF-enforced fact-checker.
3.  **[03: Multi-Agent Pipeline (LangGraph)](03_backend_pipeline.md)**
    *   Deep dive into `src/pipeline.py` and the state-based agent orchestration.
4.  **[04: AI Agents](04_agents.md)**
    *   Specialized agents for Claim Extraction, Local Knowledge Research, Web Research, and Synthesis.
5.  **[05: Models and Extraction Logic](05_models_and_extraction.md)**
    *   The Unsloth fine-tuning pipeline, the `extraction_train.jsonl` dataset, and the GGUF model.
6.  **[06: Tools and RAG Infrastructure](06_tools_and_rag.md)**
    *   Vector databases, hybrid search (Semantic + BM25), and evidence retrieval.
7.  **[07: Frontends](07_frontends.md)**
    *   Overview of the Streamlit dashboard and the React web application.
8.  **[08: Scripts and Maintenance](08_scripts_and_data.md)**
    *   Training data, fine-tuning notebooks, and database utilities.
9.  **[Quick Run Guide](quick_run_guide.md)**
    *   Step-by-step instructions to set up and launch the system.

---
*Updated on: Monday, 15 June 2026*
