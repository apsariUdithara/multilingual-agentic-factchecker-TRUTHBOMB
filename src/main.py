"""
FastAPI transport layer for the TruthBomb fact-checking engine.

Thin HTTP boundary over the LangGraph pipeline (`src.pipeline.run_fact_checker`).
The endpoint validates the request, runs the graph off the event loop, and maps
the graph's final state onto the public response schema.

Phase 5: the response now carries the canonical verdict (incl. MISLEADING),
trilingual reasoning (EN/SI/TA), a per-node audit trail, and retrieved context
built from the real re-ranked evidence.
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
    version="1.2.0",
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
    verdict: str                         # TRUE | FALSE | MISLEADING | UNVERIFIED
    confidence_score: float
    reasoning: str                       # English
    reasoning_si: str                    # Sinhala
    reasoning_ta: str                    # Tamil
    retrieved_context: List[Dict[str, Any]]
    audit_trail: List[Dict[str, Any]]
    processed_at: str


# =====================================================================
# GRAPH-STATE -> PUBLIC-SCHEMA MAPPING
# =====================================================================
_ALLOWED_WIRE_VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED"}


def _normalize_verdict(raw_verdict: Any) -> str:
    """Uppercase the canonical verdict for the wire; unknown -> UNVERIFIED."""
    verdict = str(raw_verdict).strip().upper()
    return verdict if verdict in _ALLOWED_WIRE_VERDICTS else "UNVERIFIED"


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


def _build_context(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt the re-ranked evidence into the frontend ContextSource shape."""
    context: List[Dict[str, Any]] = []
    for doc in retrieved_docs or []:
        context.append(
            {
                "content": doc.get("content", ""),
                "source": doc.get("source", "Unknown"),
                "tier": doc.get("tier", ""),
                "timestamp": doc.get("timestamp", "N/A"),
                # `distance_score` retained for frontend compatibility; now holds the
                # dense similarity (higher = closer), alongside the fused final_score.
                "distance_score": doc.get("dense_similarity", 0.0),
                "final_score": doc.get("final_score", 0.0),
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline execution failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {exc}")

    final_verdict = state.get("final_verdict") or {}
    summaries = final_verdict.get("summaries") or {}

    return VerificationResponse(
        raw_input=state.get("raw_input", payload.text),
        extracted_claim_en=state.get("extracted_claim", ""),
        verdict=_normalize_verdict(final_verdict.get("verdict", "UNVERIFIED")),
        confidence_score=_parse_confidence(final_verdict.get("confidence_score")),
        reasoning=str(
            summaries.get("english") or final_verdict.get("rationale", "No reasoning provided.")
        ).strip(),
        reasoning_si=str(summaries.get("sinhala", "")).strip(),
        reasoning_ta=str(summaries.get("tamil", "")).strip(),
        retrieved_context=_build_context(state.get("retrieved_docs", [])),
        audit_trail=state.get("audit_trail", []),
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