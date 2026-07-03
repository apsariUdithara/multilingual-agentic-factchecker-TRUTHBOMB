# Quick Run Guide: Social Media Fact-Checker

This guide provides a step-by-step procedure to set up and run the fact-checking system on your local machine.

---

## 1. Prerequisites
- **Python 3.10+**
- **Node.js & npm** (for the React frontend)
- **CUDA-compatible GPU** (highly recommended for local LLM inference)

---

## 2. Environment Setup

### Backend Setup
1. **Clone the repository** and navigate to the project root.
2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
4. **Configure Secrets**:
   - Create a `.env` file based on `.env.example`.
   - Add your `GEMINI_API_KEY` and `TAVILY_API_KEY`.
   - Add your Telegram `API_ID` and `API_HASH` if you intend to run real-time ingestion.

### Frontend Setup
1. Navigate to the `frontend/` directory:
   ```powershell
   cd frontend
   npm install
   ```

---

## 3. Model Placement
The system requires both a fine-tuned GGUF model and its corresponding LoRA adapters.
1. Create a `models/` folder in the project root if it doesn't exist.
2. **GGUF Model**: Place `phi3-extractor-unsloth.Q4_K_M.gguf` inside `models/`.
3. **LoRA Adapters**: Place the fine-tuned adapter folder (e.g., `phi3-extractor-lora`) inside `models/`.
4. **Configuration**: Update `src/config.py` to ensure `GGUF_MODEL_PATH` and `LORA_ADAPTER_PATH` point to these locations.

---

## 4. Running the System

### Step 1: Start the Backend API
From the root directory, run:
```powershell
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```
- The API will be available at `http://localhost:8000`.
- Swagger documentation can be viewed at `http://localhost:8000/docs`.

### Step 2: Start the Frontend
**Option A: React App (Recommended)**
In a new terminal, navigate to `frontend/`:
```powershell
npm run dev
```
- Access the UI at `http://localhost:5173`.

**Option B: Streamlit Dashboard**
From the root directory:
```powershell
streamlit run app.py
```
- Access the dashboard at `http://localhost:8501`.

---

## 5. Verifying the System
1. Open the frontend of your choice.
2. Paste a social media post (e.g., *"Colombo wala ada loku protest ekak thiyenawa kiyala news ekak yanawa"*).
3. Click **Verify**.
4. **Flow**:
   - The system will extract the claim into English.
   - It will check the internal ChromaDB for similar news.
   - It will perform a restricted web search on trusted sites.
   - The Synthesis Agent will provide a final verdict and multi-lingual summaries.

---

## 6. Maintenance Scripts
- **Inspect DB**: `python scripts/inspect_db.py` to see what is currently stored in the vector database.
- **Update RAG**: `python scripts/update_rag.py` to ingest new data manually.
- **Diagnose Torch**: `python scripts/diagnose_torch.py` to check GPU status.
