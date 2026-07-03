# 04: AI Agents

The `src/agents/` directory houses specialized agents that perform distinct tasks within the LangGraph ecosystem.

## 1. Claim Extractor Agent (`claim_extractor.py`)
- **Model**: Google Gemini (`gemini-2.5-flash-lite`).
- **Role**: The "Gatekeeper" and semantic analyzer.
- **Workflow**:
  - Receives raw social media text.
  - Detects the language (English, Sinhala, Tamil, or Singlish).
  - Normalizes the text into a **Professional English Declarative Claim**.
  - Routes the claim to either the **Local Researcher** (for internal OSINT data) or the **Web Researcher** (for global news).

## 2. Local Researcher Agent (`local_researcher.py`)
- **Role**: Internal Data Specialist.
- **Workflow**:
  - Receives the normalized claim.
  - Queries the local ChromaDB and BM25 indices to find historical reports or government alerts.
  - Summarizes the evidence found within the system's internal knowledge base.

## 3. Web Researcher Agent (`web_researcher.py`)
- **Model**: Google Gemini.
- **Role**: OSINT Investigator.
- **Workflow**:
  - Uses web search tools to find real-time articles and official sources.
  - Evaluates the reliability of online sources.
  - Summarizes findings to support or debunk the claim.

## 4. Synthesis Agent (`synthesis.py`)
- **Model**: Google Gemini.
- **Role**: The Final Adjudicator.
- **Workflow**:
  - Collects findings from all prior research nodes.
  - **Verdict Selection**: Assigns a verdict of TRUE, FALSE, MISLEADING, or UNVERIFIED.
  - **Multilingual Support**: Generates the final reasoning in English and provides summaries in Sinhala and Tamil.
  - **JSON Output**: Enforces a strict schema for frontend compatibility.
