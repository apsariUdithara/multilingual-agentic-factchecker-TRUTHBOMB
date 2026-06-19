# TruthBomb: Multi-Agent Social Media Fact-Checking Engine

TruthBomb is a high-performance, local-first intelligence system designed to combat misinformation on Sri Lankan social media platforms. The system is engineered to extract verifiable claims from complex, "noisy," and code-mixed linguistic streams—including English, Sinhala, Tamil, and Singlish—and verify them against authoritative Open Source Intelligence (OSINT) databases and real-time news alerts.

## Project Objectives and Vision

The primary goal of TruthBomb is to provide a robust, automated platform for the verification of social media claims. In an era of rapid information spread, the system focuses on three critical areas:
1. Linguistic Accuracy: Converting slang, emotional language, and mixed-language social media posts into professional, declarative English claims.
2. Verified Retrieval: Grounding all responses in evidence from trusted government and news sources.
3. Automated Reasoning: Using a multi-agent workflow to cross-check information from both internal archives and the broader web.

---

## Core Architectural Pillars

The system is built upon four foundational technologies that work in a synchronized pipeline to deliver accurate fact-checking results.

### 1. Linguistic Extraction (Fine-Tuned Phi-3 Model)
At the heart of the system is a Phi-3.5-mini-instruct model that has been underwent Supervised Fine-Tuning (SFT) using the Unsloth library. 
- Custom Dataset: The model was trained on a specialized dataset of instruction-input-output triplets specifically designed for the Sri Lankan linguistic context.
- Normalization Logic: It removes emotion, slang, and social media "noise," outputting a singular English declarative claim that is optimized for search engines.
- Deployment: The model is deployed locally via the GGUF format using the llama-cpp-python engine. This ensures high-speed inference and full GPU offloading on consumer-grade hardware.

**Model Fine-Tuning Notebook:**
> This Colab notebook is designed to fine-tune a pre-trained language model (specifically, unsloth/Phi-3.5-mini-instruct) for an extraction task. It then converts the fine-tuned model into a GGUF format and quantizes it to a smaller, more efficient 4-bit version for deployment. Finally, it provides options to download both the LoRA adapters and the quantized GGUF model.
> 
> 🔗 [**Access the Fine-Tuning Colab Notebook**](https://colab.research.google.com/drive/1SwNQOrSdq9DHXAcMZg0S73ZBavl3ZQYt?usp=sharing)

### 2. Multi-Agent Orchestration (LangGraph Pipeline)
TruthBomb uses LangGraph to manage a sophisticated state-based agent workflow. This allows the system to be dynamic—choosing different research paths based on the nature of the claim.
- State Persistence: The system maintains an AgentState object that tracks findings as they move through different nodes.
- Intelligent Routing: A central router analyzes the claim to decide whether to prioritize internal government databases or perform a global internet search.

### 3. Hybrid RAG Infrastructure (ChromaDB and BM25)
The system employs a "Hybrid Search" approach to evidence retrieval to ensure the highest possible recall and precision.
- Semantic Search (Dense): Uses ChromaDB with multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2) to understand the meaning behind claims, regardless of the original language.
- Lexical Search (Sparse): Uses the BM25 algorithm to perform keyword-based matching for specific entities, laws, or names.
- Ranked Context: Results from both search methods are combined and ranked to provide the most relevant evidence to the reasoning agents.

### 4. Real-Time OSINT Ingestion (Telegram MTProto)
To keep the knowledge base updated, a background worker continuously monitors official Sri Lankan news channels and government departments.
- Connectivity: Uses the Telethon library to connect directly to the Telegram MTProto API.
- Source Reliability: It polls channels like Ada Derana, News 1st, and Hiru News, ensuring the system has access to breaking news alerts before they are indexed by general search engines.
- Deduplication: Uses message identifiers to ensure that the internal database remains clean and free of redundant information.

---

## Deep Dive: The Multi-Agent Pipeline

The fact-checking process is executed by a series of specialized AI agents, each with a specific responsibility within the LangGraph ecosystem.

### Claim Extractor Agent
This agent acts as the system's "Gatekeeper." It analyzes the raw social media text to detect the language and extract the core claim. Most importantly, it performs a classification task to determine if the claim should be routed to local research or web research.

### Local Researcher Agent
This agent is a specialist in the system's internal knowledge base. It uses the local database tools to find historical reports, government gazettes, and official alerts stored in the vector database. It is optimized for finding "ground truth" in local records.

### Web Researcher Agent
When a claim involves international news or real-time events not yet in the local archive, this agent performs a restricted web search. It is configured to only trust a whitelist of authoritative domains, preventing the ingestion of further misinformation.

### Synthesis Agent
The final node in the graph, the Synthesis Agent, collects findings from all previous nodes. It cross-references the evidence, identifies contradictions, and generates the final verdict. It also generates reasoning summaries in English, Sinhala, and Tamil for maximum accessibility.

---

## Technical Stack and Dependencies

- Programming Language: Python 3.10 or higher.
- API Framework: FastAPI for high-performance REST services.
- Agent Framework: LangGraph for stateful multi-agent orchestration.
- ML Inference: llama-cpp-python (GGUF Support) and Transformers (PEFT/LoRA).
- Databases: ChromaDB (Vector Store) and Rank-BM25 (Lexical).
- Frontend: React with TypeScript and Vite for a modern web interface.
- Style: Tailwind CSS for responsive and clean design.

---

## Prerequisites and Environment Configuration

Before running the system, ensure you have the following installed:
- Node.js (version 20 or higher) and npm.
- Python 3.10 (with a virtual environment recommended).
- A CUDA-compatible GPU (highly recommended for local LLM inference).

### Environment Variables (.env)
You must create a .env file in the root directory. Refer to the .env.example file for the required keys.
- GEMINI_API_KEY: Required for the agentic orchestration.
- TAVILY_API_KEY: Required for the web research tool.
- TELEGRAM_API_ID and TELEGRAM_API_HASH: Required for real-time ingestion.

---

## Installation and Initialization

1. Dependency Installation:
   Run the following command in the root directory to install the Python stack:
   pip install -r requirements.txt

2. Frontend Setup:
   Navigate to the frontend directory and install the Node packages:
   cd frontend
   npm install

3. Model Placement:
   Place your fine-tuned GGUF model and LoRA adapter folders in the models/ directory.
   - GGUF: models/phi3-extractor-unsloth.Q4_K_M.gguf
   - LoRA: models/phi3-extractor-lora/

---

## Running the System

### Option 1: One-Click Startup (Recommended)
We have provided a self-healing PowerShell script that automates the startup process. It checks for missing dependencies, installs them if necessary, and launches both the backend and frontend.
Execute the following in your terminal:
.\start.ps1

### Option 2: Manual Execution
If you prefer to run the components individually:
- Backend: uvicorn src.main:app --port 8000
- Frontend: cd frontend; npm run dev

Access the Web Application at http://localhost:5173 and the API documentation at http://localhost:8000/docs.

---

## Maintenance and Diagnostics

The system includes several scripts for database and environment maintenance:
- scripts/inspect_db.py: Use this to check the current contents of your vector database.
- scripts/ingest.py: Use this to manually ingest new documents or local datasets.
- scripts/diagnose_torch.py: Ensures your GPU is correctly configured for local inference.

---

## Project Directory Map

- src/main.py: The main FastAPI entry point and Telegram background worker.
- src/pipeline.py: The LangGraph workflow and state definitions.
- src/agents/: Contains the individual logic for each AI agent.
- src/tools/: Contains the retrieval and search tools.
- src/models/: Contains the local LLM wrapper classes.
- pipeline-setup/vector_db/: The persistent storage for ChromaDB.
- docs/: (Local Only) Detailed technical modules and architectural notes.
- models/: (Local Only) The local weights for the fine-tuned models.

---

## Final Note on Documentation and Models
Please note that the /docs folder and the /models folder are intended to remain local to your machine to ensure privacy and efficiency. The README serves as the primary comprehensive guide for the repository.
