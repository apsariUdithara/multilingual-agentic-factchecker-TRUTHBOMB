"""
FastAPI transport layer for the TruthBomb fact-checking engine.

Architecture note (Phase 1 — single canonical pipeline):
    This module is now a *thin* HTTP boundary. All verification logic lives in the
    LangGraph pipeline (``src.pipeline.run_fact_checker``). The endpoint's only jobs
    are: (1) validate the request, (2) run the graph off the event loop, and
    (3) map the graph's final state onto the public response schema.

    Removed in Phase 1:
        - the inline retrieval + ``execute_fact_checker_llm`` path (duplicated the graph)
        - the in-process Telegram ingestion worker and its hardcoded credentials
          (ingestion now runs standalone via ``scripts/update_rag_telegram.py``)
        - eager startup loading of embeddings / vector store / extractor model
          (the graph's agents lazy-load their own resources on first use)

    Not changed in Phase 1 (deliberately deferred):
        - the public response schema and ``frontend/types.ts`` (widened in Phase 5)
        - the extractor model, routing strategy, and re-ranking (Phases 2-4)
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import run_fact_checker
from src.config import VECTOR_DB_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TruthBomb Fact-Checking Engine",
    description="Thin HTTP boundary over the LangGraph multi-agent verification pipeline.",
    version="1.1.0",
)


# =====================================================================
# DATA TRANSFER OBJECTS (SCHEMAS)
# =====================================================================
class VerificationRequest(BaseModel):
    text: str
    metadata_hint: Optional[str] = None


class VerificationResponse(BaseModel):
    raw_input: str
    extracted_claim_en: str
    verdict: str
    confidence_score: float
    retrieved_context: List[Dict[str, Any]]
    reasoning: str
    processed_at: str


# =====================================================================
# GRAPH-STATE -> PUBLIC-SCHEMA MAPPING
# ---------------------------------------------------------------------
# The LangGraph synthesis node emits a richer verdict vocabulary
# ("True"/"False"/"Misleading"/"Unverified"), a percentage-string
# confidence, and EN/SI/TA summaries. The current public schema (and the
# frontend) only model a 3-value verdict + float confidence + English
# reasoning, so we adapt here. The richer fields are surfaced in Phase 5.
# =====================================================================
_VERDICT_MAP = {
    "TRUE": "TRUE",
    "FALSE": "FALSE",
    "MISLEADING": "UNVERIFIED",
    "UNVERIFIED": "UNVERIFIED",
}


def _normalize_verdict(raw_verdict: Any) -> str:
    """Collapse the graph's verdict vocabulary onto the public 3-value enum."""
    return _VERDICT_MAP.get(str(raw_verdict).strip().upper(), "UNVERIFIED")


def _parse_confidence(raw_confidence: Any) -> float:
    """Accept '85%', '0.85', 85, or 0.85 and return a float clamped to [0, 1]."""
    if raw_confidence is None:
        return 0.0
    text = str(raw_confidence).strip().rstrip("%").strip()
    try:
        value = float(text)
    except (TypeError, ValueError):
        return 0.0
    if value > 1.0:  # value was expressed as a percentage
        value /= 100.0
    return max(0.0, min(1.0, value))


def _build_context(agent_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adapt the graph's per-agent findings into the ContextSource shape the frontend
    expects. Phase 1 surfaces each agent's textual finding; structured per-document
    retrieval scores arrive with the dedicated retrieve node in Phase 3.
    """
    context: List[Dict[str, Any]] = []
    for output in agent_outputs or []:
        context.append(
            {
                "content": output.get("findings", ""),
                "distance_score": 0.0,
                "source": f"{output.get('agent', 'unknown')} [{output.get('source_type', 'n/a')}]",
                "timestamp": "N/A",
            }
        )
    return context


# =====================================================================
# API ROUTING LOGIC
# =====================================================================
@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify_claim(payload: VerificationRequest) -> VerificationResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text stream cannot be empty.")

    try:
        # run_fact_checker is synchronous and performs blocking model / network
        # calls; run it in a worker thread so the event loop stays responsive.
        state = await asyncio.to_thread(run_fact_checker, payload.text)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure as a 500
        logger.exception("Pipeline execution failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {exc}")

    final_verdict = state.get("final_verdict") or {}

    return VerificationResponse(
        raw_input=state.get("raw_input", payload.text),
        extracted_claim_en=state.get("extracted_claim", ""),
        verdict=_normalize_verdict(final_verdict.get("verdict", "UNVERIFIED")),
        confidence_score=_parse_confidence(final_verdict.get("confidence_score")),
        retrieved_context=_build_context(state.get("agent_outputs", [])),
        reasoning=str(final_verdict.get("rationale", "No reasoning provided.")).strip(),
        processed_at=datetime.utcnow().isoformat(),
    )


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """Cheap liveness probe: graph is importable and the vector store exists on disk."""
    db_path = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
    db_present = os.path.exists(db_path)
    return {
        "status": "healthy" if db_present else "degraded",
        "graph_loaded": callable(run_fact_checker),
        "vector_store_present": db_present,
        "timestamp": datetime.utcnow().isoformat(),
    }