"""Offline retrieval-quality evaluation for the TruthBomb RAG agent.

Uses the dataset's own structure as ground truth: every post is a query, and a
retrieved chunk is *relevant* if it carries the same ``event_id`` as the query's
event. No manual labelling required.

Reports Recall@k, MRR and nDCG@k for the hybrid retriever, so any change to the
embedding model, fusion or reranker can be measured before/after.

Prerequisite: populate the store first with ``python -m scripts.update_rag``.
Run:  ``python -m tests.eval_retrieval``
"""
import json
import logging
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME
from src.rag.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = "data/dataset.json"
K = 5


def build_retriever() -> HybridRetriever:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME)
    data = db.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m or {})
            for t, m in zip(data.get("documents", []), data.get("metadatas", []))]
    if not docs:
        raise SystemExit("Vector store is empty. Run: python -m scripts.update_rag")
    bm25 = BM25Retriever.from_documents(docs)
    return HybridRetriever(db, bm25_retriever=bm25)


def evaluate() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    retriever = build_retriever()

    recall_hits, rr_sum, ndcg_sum, n = 0, 0.0, 0.0, 0
    for entry in dataset:
        event_id = entry.get("claim_id") or entry.get("core_event", "")[:40]
        for var in entry.get("variations", []):
            query = var.get("raw_post", "")
            if not query.strip():
                continue
            n += 1
            results = retriever.retrieve(query, top_n=K)
            rels = [1 if r["doc"].metadata.get("event_id") == event_id else 0 for r in results]

            if any(rels):
                recall_hits += 1
                first = rels.index(1)
                rr_sum += 1.0 / (first + 1)
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(sum(rels), K))) or 1.0
            ndcg_sum += dcg / idcg

    if n == 0:
        logger.error("No queries evaluated.")
        return
    logger.info("Queries evaluated: %d", n)
    logger.info("Recall@%d : %.4f", K, recall_hits / n)
    logger.info("MRR@%d    : %.4f", K, rr_sum / n)
    logger.info("nDCG@%d   : %.4f", K, ndcg_sum / n)


if __name__ == "__main__":
    evaluate()
