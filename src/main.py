import logging
import json
import re
import asyncio
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from telethon import TelegramClient

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.models.extractor import LinguisticExtractor
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME, TARGET_CHANNELS

try:
    from llama_cpp import LlamaGrammar
except ImportError:
    LlamaGrammar = None

# --- TELEGRAM CREDENTIALS ---
API_ID = 38020229
API_HASH = 'bad58fa93d224c8320db9639efa7ed5e'
SESSION_NAME = os.path.join("D:/7th/AI/Project/scripts", 'rag_updater_session')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Claim Fact-Checking Engine",
    description="Automated system to extract claims from mixed linguistic streams and verify them against live OSINT databases.",
    version="1.0.0"
)

embeddings = None
vector_db = None
extractor_model = None
bm25_retriever = None

# GBNF Grammar strictly enforcing valid JSON output for the fact-checker
GBNF_GRAMMAR_STR = r'''
root ::= "{" ws "\"verdict\"" ws ":" ws verdict ws "," ws "\"confidence_score\"" ws ":" ws number ws "," ws "\"reasoning\"" ws ":" ws string "}"
verdict ::= "\"TRUE\"" | "\"FALSE\"" | "\"UNVERIFIED\""
number ::= "0." [0-9]+ | "1.0" | "0.0" | "0" | "1"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
'''

async def telegram_ingestion_worker():
    logger.info("Starting Telegram MTProto background polling...")
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    
    while True:
        try:
            if vector_db is None:
                await asyncio.sleep(10)
                continue
                
            current_data = vector_db.get(include=["metadatas"])
            existing_ids = set()
            if current_data and "metadatas" in current_data:
                existing_ids = {
                    meta.get("message_id") 
                    for meta in current_data["metadatas"] 
                    if meta and "message_id" in meta
                }

            new_documents = []
            for channel in TARGET_CHANNELS:
                async for message in client.iter_messages(channel, limit=20):
                    if not message.text:
                        continue
                    msg_id = str(message.id)
                    if msg_id in existing_ids:
                        continue
                    
                    doc = Document(
                        page_content=f"Source: {channel}\nAlert: {message.text}",
                        metadata={
                            "source": f"Telegram-@{channel}",
                            "message_id": msg_id,
                            "timestamp": message.date.isoformat() if message.date else datetime.utcnow().isoformat(),
                            "ingested_at": datetime.utcnow().isoformat()
                        }
                    )
                    new_documents.append(doc)
                    existing_ids.add(msg_id)
            
            if new_documents:
                logger.info(f"Injecting {len(new_documents)} new social media alerts into ChromaDB...")
                vector_db.add_documents(new_documents)
                update_bm25_retriever()
        except Exception as e:
            logger.error(f"Telegram polling error: {e}")
            
        await asyncio.sleep(60)

def update_bm25_retriever():
    global bm25_retriever
    try:
        if vector_db:
            all_data = vector_db.get(include=["documents", "metadatas"])
            if all_data and all_data.get("documents"):
                docs = []
                for text, meta in zip(all_data["documents"], all_data["metadatas"]):
                    docs.append(Document(page_content=text, metadata=meta or {}))
                if docs:
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    logger.info(f"BM25 index updated with {len(docs)} documents.")
    except Exception as e:
        logger.error(f"Failed to update BM25 retriever: {e}")

@app.on_event("startup")
async def startup_event():
    global embeddings, vector_db, extractor_model
    logger.info("Bootstrapping localized embedding models and vector infrastructure...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        logger.info("ChromaDB state successfully mounted and pooled.")
        update_bm25_retriever()
    except Exception as e:
        logger.critical(f"Failed to bind vector database instance: {e}")
        raise RuntimeError(f"Database Initialization Defect: {e}")

    try:
        extractor_model = LinguisticExtractor()
        logger.info("LinguisticExtractor loaded successfully.")
    except Exception as e:
        extractor_model = None
        logger.exception("Failed to load LinguisticExtractor: %s", e)

    # Start Telegram background ingestion
    asyncio.create_task(telegram_ingestion_worker())

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

def execute_fact_checker_llm(claim: str, context_documents: List[str]) -> Dict[str, Any]:
    if not context_documents:
        return {
            "verdict": "UNVERIFIED",
            "confidence_score": 0.0,
            "reasoning": "No supporting context documents were retrieved for this claim."
        }

    if extractor_model is None or not hasattr(extractor_model, "llm"):
        raise RuntimeError("Fact-checking model is not available.")

    context_blob = "\n\n".join(context_documents)
    prompt = (
        "You are a strict fact-checking engine.\n"
        "Task: Determine whether the claim is TRUE, FALSE, or UNVERIFIED based strictly on the provided context.\n"
        "Rules:\n"
        "- Use ONLY the context. Do not use outside knowledge.\n"
        "- If context is insufficient or contradictory, verdict must be UNVERIFIED.\n"
        "- Return ONLY valid JSON, no markdown, no prose.\n"
        "- JSON schema: {\"verdict\":\"TRUE|FALSE|UNVERIFIED\",\"confidence_score\":float,\"reasoning\":\"string\"}.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Context:\n{context_blob}\n"
    )

    grammar = None
    if LlamaGrammar is not None:
        try:
            grammar = LlamaGrammar.from_string(GBNF_GRAMMAR_STR)
        except Exception as e:
            logger.warning(f"Failed to load GBNF grammar: {e}")

    raw = extractor_model.llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        echo=False,
        grammar=grammar
    )
    raw_text = raw["choices"][0]["text"].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            raise RuntimeError(f"Fact-checker returned non-JSON output: {raw_text}")
        parsed = json.loads(match.group(0))

    verdict = str(parsed.get("verdict", "UNVERIFIED")).upper()
    if verdict not in {"TRUE", "FALSE", "UNVERIFIED"}:
        verdict = "UNVERIFIED"

    confidence_raw = parsed.get("confidence_score", 0.0)
    try:
        confidence_score = float(confidence_raw)
    except (TypeError, ValueError):
        confidence_score = 0.0
    confidence_score = max(0.0, min(1.0, confidence_score))

    reasoning = str(parsed.get("reasoning", "No reasoning provided by model.")).strip()

    return {
        "verdict": verdict,
        "confidence_score": confidence_score,
        "reasoning": reasoning,
    }

# =====================================================================
# API ROUTING LOGIC
# =====================================================================
@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify_claim(payload: VerificationRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text stream cannot be empty.")

    if extractor_model is None:
        raise HTTPException(status_code=503, detail="Extractor model is not loaded. Check server logs.")
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database is not ready.")

    try:
        # Step 1: Linguistic Extraction
        extracted_claim = extractor_model.extract_claim(payload.text)
        logger.info(f"Normalized Semantic Claim: {extracted_claim}")

        # Step 2: Hybrid Context Retrieval (ChromaDB + BM25)
        matched_docs = vector_db.similarity_search_with_score(query=extracted_claim, k=3)
        
        hybrid_docs = {}
        for doc, score in matched_docs:
            hybrid_docs[doc.page_content] = {"doc": doc, "score": float(score), "source": "chroma"}

        if bm25_retriever:
            bm25_results = bm25_retriever.invoke(extracted_claim)[:2]
            for doc in bm25_results:
                if doc.page_content not in hybrid_docs:
                    hybrid_docs[doc.page_content] = {"doc": doc, "score": 1.0, "source": "bm25"}

        formatted_context = []
        raw_context_texts = []
        
        for content, data in hybrid_docs.items():
            doc = data["doc"]
            raw_context_texts.append(content)
            formatted_context.append({
                "content": content,
                "distance_score": data["score"],
                "source": f"{doc.metadata.get('source', 'Unknown')} [{data['source']}]",
                "timestamp": doc.metadata.get("timestamp", "N/A")
            })
            
        logger.info(f"Extracted {len(formatted_context)} relevant context nodes from hybrid search.")

        # Step 3: Fact-Checking Analysis (LLM with GBNF)
        analysis = execute_fact_checker_llm(extracted_claim, raw_context_texts)

        return VerificationResponse(
            raw_input=payload.text,
            extracted_claim_en=extracted_claim,
            verdict=analysis["verdict"],
            confidence_score=analysis["confidence_score"],
            retrieved_context=formatted_context,
            reasoning=analysis["reasoning"],
            processed_at=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Execution failure during request processing pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Processing Exception: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    is_ready = vector_db is not None
    return {
        "status": "healthy" if is_ready else "unready",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": is_ready
    }
