"""Unified hybrid retriever for the TruthBomb RAG agent.

Before this module there were two divergent retrieval code paths:

* ``src/main.py`` — dense Chroma + BM25 merged by hand (flat BM25 scores).
* ``src/tools/local_db_tool.py`` — a plain ``similarity_search(k=3)``, no sparse
  search and no fusion.

They could return different evidence for the same claim. ``HybridRetriever`` is
now the single retrieval implementation both call, so behaviour is consistent:

    dense (Chroma) + sparse (BM25)  ->  Reciprocal Rank Fusion  ->  cross-encoder rerank

The reranker is optional and fails safe (see ``fusion.py``).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.config import (
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_TOP_N,
    RRF_K,
    ENABLE_RERANKER,
    RERANKER_MODEL_NAME,
)
from src.rag.fusion import reciprocal_rank_fusion, CrossEncoderReranker

logger = logging.getLogger(__name__)

# One shared reranker instance (loads its model lazily, at most once).
_reranker = CrossEncoderReranker(RERANKER_MODEL_NAME, enabled=ENABLE_RERANKER)


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion and optional re-ranking."""

    def __init__(self, vector_db, bm25_retriever=None, reranker: Optional[CrossEncoderReranker] = None):
        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker or _reranker

    def set_bm25(self, bm25_retriever) -> None:
        """Swap in a freshly rebuilt BM25 index (called after ingestion)."""
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        candidates: int = RETRIEVAL_CANDIDATES,
        top_n: int = RETRIEVAL_TOP_N,
    ) -> List[Dict]:
        """Return the most relevant chunks for ``query``.

        Each item is ``{"doc", "sources", "fused_score"}`` where ``sources`` lists
        which retrievers ("dense"/"sparse") surfaced the chunk.
        """
        if not query or not query.strip():
            return []

        # 1. Dense candidates (Chroma).
        dense_docs: List = []
        try:
            dense_docs = self.vector_db.similarity_search(query, k=candidates)
        except Exception as exc:  # pragma: no cover - depends on live DB
            logger.error("Dense retrieval failed: %s", exc)

        # 2. Sparse candidates (BM25).
        sparse_docs: List = []
        if self.bm25_retriever is not None:
            try:
                sparse_docs = self.bm25_retriever.invoke(query)[:candidates]
            except Exception as exc:  # pragma: no cover
                logger.error("Sparse retrieval failed: %s", exc)

        # Map content -> doc, and track which retriever(s) found it.
        by_key: Dict[str, object] = {}
        found_in: Dict[str, set] = {}
        for tag, docs in (("dense", dense_docs), ("sparse", sparse_docs)):
            for d in docs:
                key = d.page_content
                by_key.setdefault(key, d)
                found_in.setdefault(key, set()).add(tag)

        if not by_key:
            return []

        # 3. Reciprocal Rank Fusion over the two ranked lists.
        fused = reciprocal_rank_fusion(
            [[d.page_content for d in dense_docs], [d.page_content for d in sparse_docs]]
        )

        fused_docs = [by_key[key] for key, _ in fused if key in by_key]
        fused_scores = {key: score for key, score in fused}

        # 4. Optional cross-encoder re-ranking of the fused shortlist.
        reranked = self.reranker.rerank(query, fused_docs, top_n=top_n)

        results: List[Dict] = []
        for d in reranked:
            key = d.page_content
            results.append(
                {
                    "doc": d,
                    "sources": sorted(found_in.get(key, {"dense"})),
                    "fused_score": float(fused_scores.get(key, 0.0)),
                }
            )
        return results
