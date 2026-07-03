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
from src.rag.retriever import HybridRetriever
from src.config import (
    VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME, TARGET_CHANNELS,
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_NAME,
    RETRIEVAL_CANDIDATES, RETRIEVAL_TOP_N,
)
from src.pipeline import run_fact_checker

try:
    from llama_cpp import LlamaGrammar
except ImportError:
    LlamaGrammar = None

# --- TELEGRAM CREDENTIALS (loaded from .env via src.config; never hard-coded) ---
API_ID = TELEGRAM_API_ID
API_HASH = TELEGRAM_API_HASH
SESSION_NAME = TELEGRAM_SESSION_NAME

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
retriever = None  # HybridRetriever: dense + sparse + RRF + rerank

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
    
    is_first_run = True
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
            # Backfill more history (150 messages) on first boot to cover the past week of news,
            # then use a lightweight polling limit of 20 messages.
            fetch_limit = 150 if is_first_run else 20
            
            for channel in TARGET_CHANNELS:
                async for message in client.iter_messages(channel, limit=fetch_limit):
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
                
            is_first_run = False
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
                    if retriever is not None:
                        retriever.set_bm25(bm25_retriever)
                    logger.info(f"BM25 index updated with {len(docs)} documents.")
    except Exception as e:
        logger.error(f"Failed to update BM25 retriever: {e}")

@app.on_event("startup")
async def startup_event():
    global embeddings, vector_db, extractor_model, retriever
    logger.info("Bootstrapping localized embedding models and vector infrastructure...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        logger.info("ChromaDB state successfully mounted and pooled.")
        retriever = HybridRetriever(vector_db)
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
    routing_destination: Optional[str] = None
    summaries: Optional[Dict[str, str]] = None

def execute_fact_checker_llm(claim: str, context_documents: List[str]) -> Dict[str, Any]:
    if not context_documents:
        return {
            "verdict": "UNVERIFIED",
            "confidence_score": 0.0,
            "reasoning": "No supporting context documents were retrieved for this claim."
        }

    context_blob = "\n\n".join(context_documents)
    
    # 1. Try strict verification using the Gemini API to prevent local model hallucinations
    try:
        import google.generativeai as genai
        from src.config import GEMINI_API_KEY
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")
            
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = (
            "You are a strict fact-checking engine.\n"
            "Task: Determine whether the claim is TRUE, FALSE, or UNVERIFIED based strictly on the provided context.\n"
            "Rules:\n"
            "- Use ONLY the provided context. Do not use outside knowledge.\n"
            "- If the context does not explicitly confirm or refute the claim, the verdict MUST be UNVERIFIED.\n"
            "- Do not make assumptions or extrapolate. If the documents talk about completely different topics, verdict is UNVERIFIED.\n"
            "- Return ONLY valid JSON, no markdown, no prose.\n"
            "- JSON schema: {\"verdict\":\"TRUE|FALSE|UNVERIFIED\",\"confidence_score\":float,\"reasoning\":\"string\"}.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Context:\n{context_blob}\n"
        )
        
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip().replace('```json', '').replace('```', '')
        
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if not match:
                raise RuntimeError(f"Gemini returned non-JSON output: {raw_text}")
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
        
        reasoning = str(parsed.get("reasoning", "No reasoning provided.")).strip()
        
        return {
            "verdict": verdict,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
        }
    except Exception as e:
        logger.warning(f"Gemini fact-checking failed (falling back to local GGUF model): {e}")

    # 2. Fallback to Local GGUF if Gemini fails or is offline
    if extractor_model is None or not hasattr(extractor_model, "llm"):
        raise RuntimeError("Fact-checking model is not available.")

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

    try:
        # Run the multi-agent LangGraph workflow
        logger.info(f"Invoking multi-agent LangGraph pipeline for input: {payload.text[:100]}...")
        result = await asyncio.to_thread(run_fact_checker, payload.text)
        
        extracted_claim = result.get("extracted_claim", "")
        routing_destination = result.get("routing_destination", "")
        agent_outputs = result.get("agent_outputs", [])
        final_verdict = result.get("final_verdict", {})
        
        # Format the context findings from active research agents
        formatted_context = []
        for out in agent_outputs:
            formatted_context.append({
                "content": out.get("findings", ""),
                "distance_score": 0.0,
                "source": f"{out.get('agent', 'Researcher')} [{out.get('source_type', 'unknown')}]",
                "timestamp": datetime.utcnow().isoformat(),
            })

        verdict = str(final_verdict.get("verdict", "UNVERIFIED")).upper()
        
        conf_str = str(final_verdict.get("confidence_score", "0%"))
        try:
            confidence_score = float(conf_str.replace("%", "").strip()) / 100.0
        except Exception:
            confidence_score = 0.0
            
        reasoning = str(final_verdict.get("rationale", "No explanation provided.")).strip()
        summaries = final_verdict.get("summaries", {
            "english": reasoning,
            "sinhala": "තොරතුරු නොමැත",
            "tamil": "தகவல் இல்லை"
        })

        return VerificationResponse(
            raw_input=payload.text,
            extracted_claim_en=extracted_claim,
            verdict=verdict,
            confidence_score=confidence_score,
            retrieved_context=formatted_context,
            reasoning=reasoning,
            processed_at=datetime.utcnow().isoformat(),
            routing_destination=routing_destination,
            summaries=summaries
        )

    except Exception as e:
        logger.error(f"Execution failure during multi-agent pipeline verification: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Processing Exception: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    is_ready = vector_db is not None
    return {
        "status": "healthy" if is_ready else "unready",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": is_ready
    }
