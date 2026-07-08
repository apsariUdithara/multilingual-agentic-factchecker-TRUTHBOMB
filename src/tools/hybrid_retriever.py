"""
Hybrid evidence retriever for the TruthBomb pipeline (Phase 3).

Combines dense (Chroma / multilingual MiniLM) and lexical (BM25) retrieval, fuses
the two rankings with Reciprocal Rank Fusion (RRF), then re-ranks the fused set by
source tier and recency. Fusion and re-ranking are fully deterministic, and every
component score is exposed per document for the audit trail.

The Chroma store and embeddings are shared with `local_db_tool` (the same lazy
singletons), so the process holds a single connection and a single embedding model.

Freshness note: the dense store is queried live, but the BM25 index is built once
from the corpus on first use and cached for the process lifetime. Because ingestion
runs in a separate worker process, documents added after startup appear in dense
results immediately but only enter BM25 after an API restart.
"""

from datetime import datetime, timezone
from typing import List, Tuple, Dict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import (
    RETRIEVAL_K_DENSE,
    RETRIEVAL_K_LEXICAL,
    RETRIEVAL_K_FINAL,
    RRF_K,
    TRUSTED_SOURCE_IDENTIFIERS,
    TIER_WEIGHT_TRUSTED,
    TIER_WEIGHT_UNTRUSTED,
    RECENCY_WEIGHTS,
    RECENCY_WEIGHT_STALE,
    RECENCY_WEIGHT_UNKNOWN,
)
from src.tools.local_db_tool import get_vector_db


def _doc_key(doc: Document) -> str:
    """Stable identity for de-duplicating a document across the two retrievers."""
    return str((doc.metadata or {}).get("message_id") or doc.page_content[:200])


def tier_of(source: str) -> str:
    """Classify a document's source as 'trusted' or 'untrusted'."""
    src = (source or "").lower()
    return "trusted" if any(ident.lower() in src for ident in TRUSTED_SOURCE_IDENTIFIERS) else "untrusted"


def _tier_weight(source: str) -> float:
    return TIER_WEIGHT_TRUSTED if tier_of(source) == "trusted" else TIER_WEIGHT_UNTRUSTED


def _recency_weight(timestamp: str) -> float:
    """Bucketed recency weight from an ISO timestamp; neutral if missing/unparseable."""
    if not timestamp:
        return RECENCY_WEIGHT_UNKNOWN
    try:
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return RECENCY_WEIGHT_UNKNOWN
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - dt).days
    for max_age, weight in RECENCY_WEIGHTS:
        if age_days <= max_age:
            return weight
    return RECENCY_WEIGHT_STALE


class HybridRetriever:
    """Dense + lexical retrieval fused by RRF and re-ranked by tier and recency."""

    def __init__(self):
        self.vector_db = get_vector_db()
        self._bm25 = None
        self._bm25_built = False

    def _get_bm25(self):
        """Build the BM25 index from the current corpus once, then cache it."""
        if not self._bm25_built:
            self._bm25_built = True
            data = self.vector_db.get(include=["documents", "metadatas"])
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            corpus = [
                Document(page_content=content, metadata=meta or {})
                for content, meta in zip(documents, metadatas)
                if content
            ]
            if corpus:
                self._bm25 = BM25Retriever.from_documents(corpus)
                self._bm25.k = RETRIEVAL_K_LEXICAL
        return self._bm25

    def retrieve(self, claim: str) -> Tuple[List[Dict], float]:
        """
        Return (re-ranked documents, retrieval_score).

        retrieval_score is the best dense similarity found in the KB (in (0, 1]),
        i.e. how well our strongest local evidence matches the claim. It is the
        signal the score-based router will threshold in Phase 4.
        """
        if not claim or not claim.strip():
            return [], 0.0

        # 1. Dense retrieval: convert raw distance -> bounded similarity in (0, 1].
        dense_hits = self.vector_db.similarity_search_with_score(claim, k=RETRIEVAL_K_DENSE)
        dense_docs = [doc for doc, _ in dense_hits]
        dense_sim = {_doc_key(doc): 1.0 / (1.0 + dist) for doc, dist in dense_hits}

        # 2. Lexical retrieval (used for rank position only).
        bm25 = self._get_bm25()
        bm25_docs = bm25.invoke(claim) if bm25 is not None else []

        # 3. Reciprocal Rank Fusion across both rankings.
        docs_by_key: Dict[str, Document] = {}
        rrf: Dict[str, float] = {}
        for ranking in (dense_docs, bm25_docs):
            for rank, doc in enumerate(ranking):
                key = _doc_key(doc)
                docs_by_key.setdefault(key, doc)
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        # 4. Re-rank multiplicatively by tier and recency; expose all components.
        scored: List[Dict] = []
        for key, doc in docs_by_key.items():
            meta = doc.metadata or {}
            source = meta.get("source", "")
            timestamp = meta.get("timestamp", "")
            tw = _tier_weight(source)
            rw = _recency_weight(timestamp)
            base = rrf[key]
            scored.append(
                {
                    "content": doc.page_content,
                    "source": source,
                    "tier": tier_of(source),
                    "timestamp": timestamp,
                    "dense_similarity": round(dense_sim.get(key, 0.0), 4),
                    "rrf_score": round(base, 6),
                    "tier_weight": tw,
                    "recency_weight": rw,
                    "final_score": round(base * tw * rw, 6),
                }
            )

        # Deterministic ordering: final score, then dense similarity, then source.
        scored.sort(
            key=lambda d: (d["final_score"], d["dense_similarity"], d["source"]),
            reverse=True,
        )
        top = scored[:RETRIEVAL_K_FINAL]

        retrieval_score = max(dense_sim.values(), default=0.0)
        return top, round(retrieval_score, 4)


# --- Lazy singleton --------------------------------------------------------
_retriever_instance = None


def get_hybrid_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance