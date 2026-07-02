"""Rank fusion and re-ranking utilities for the TruthBomb RAG agent.

Previously ``main.py`` merged dense (Chroma) and sparse (BM25) hits by hand and
gave every BM25 result a flat score of ``1.0``. Dense *distances* and sparse
*ranks* live on completely different scales, so that "fusion" was effectively
random for the sparse half.

This module replaces that with two standard, defensible techniques:

* **Reciprocal Rank Fusion (RRF)** — scale-free; only ranks matter, so dense and
  sparse lists combine cleanly.
* An optional **cross-encoder re-ranker** — a second-stage model that scores
  ``(query, document)`` pairs directly for relevance. It degrades gracefully to
  the RRF order when the model or its dependencies are unavailable.

The RRF core has no heavy dependencies so it can be unit-tested on any machine.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> List[Tuple[str, float]]:
    """Fuse several ranked lists of ids into one, using RRF.

    Args:
        ranked_lists: each inner sequence is a list of document keys ordered
            best-first (e.g. the page_content of dense hits, then of sparse hits).
        k: RRF damping constant. Larger ``k`` flattens the contribution of top
            ranks; 60 is the canonical default from Cormack et al. (2009).
        weights: optional per-list weights (same length as ``ranked_lists``).

    Returns:
        ``(key, score)`` tuples sorted by descending fused score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")

    scores: Dict[str, float] = {}
    for lst, weight in zip(ranked_lists, weights):
        for rank, key in enumerate(lst):
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class CrossEncoderReranker:
    """Lazy, fail-safe wrapper around a sentence-transformers CrossEncoder.

    The model is loaded on first use. If ``sentence-transformers`` is missing or
    the weights cannot be fetched, the reranker disables itself and callers fall
    back to the fusion order — the pipeline never crashes because of it.
    """

    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._failed = False

    def _ensure_model(self):
        if self._model is not None or self._failed or not self.enabled:
            return
        try:
            from sentence_transformers import CrossEncoder  # heavy, optional
            logger.info("Loading cross-encoder reranker: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.warning(
                "Reranker unavailable (%s); falling back to RRF ordering.", exc
            )
            self._failed = True

    def rerank(
        self,
        query: str,
        documents: List,
        top_n: int,
        text_getter: Callable[[object], str] = lambda d: getattr(d, "page_content", str(d)),
    ) -> List:
        """Return the ``top_n`` documents most relevant to ``query``.

        Falls back to the input order (truncated to ``top_n``) whenever the model
        is disabled or fails to load.
        """
        if not documents:
            return []
        self._ensure_model()
        if self._model is None:
            return documents[:top_n]
        try:
            pairs = [(query, text_getter(d)) for d in documents]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda ds: ds[1], reverse=True)
            return [d for d, _ in ranked[:top_n]]
        except Exception as exc:  # pragma: no cover
            logger.warning("Reranking failed (%s); using RRF ordering.", exc)
            return documents[:top_n]
