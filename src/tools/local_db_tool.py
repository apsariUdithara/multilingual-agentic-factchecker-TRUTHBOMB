"""
Shared vector-store access for the pipeline.

After Phase 4 the local verifier runs on the GGUF model, so the previous
LoRA/transformers reasoning path (and its torch/peft imports) has been removed.
This module now only owns the lazy Chroma + embeddings singletons that the
hybrid retriever depends on.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

_embeddings = None
_vector_db = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=get_embeddings(),
            collection_name=CHROMA_COLLECTION_NAME,
        )
    return _vector_db