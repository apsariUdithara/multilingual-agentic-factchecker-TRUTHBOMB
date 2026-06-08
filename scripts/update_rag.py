import os
import time
import logging
from curl_cffi import requests
import feedparser
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================================
# SYSTEM CONFIGURATION
# =====================================================================
VECTOR_DB_DIR = "D:/7th/AI/Project/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "traffic_telemetry"

# Setup structured console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RAGDatabaseUpdater:
    def __init__(self):
        logger.info("Initializing RAG Database continuous ingestion pipeline...")
        
        # Enforce HTTPS paths to eliminate silent redirect dropouts
        self.sources = {
            "Ada Derana": "https://www.adaderana.lk/rss.php",
            "NewsFirst English": "https://english.newsfirst.lk/feed"
        }
        
        # Initialize text embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Reference persistent Chroma instance
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        
        # Initialize an authoritative browser session
        self.session = requests.Session()
        self._initialize_session_headers()

    def _initialize_session_headers(self):
        """
        Populates a complete browser footprint to satisfy Cloudflare 
        and enterprise firewall identity validation checks.
        """
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })

    def _get_existing_links(self) -> set:
        """
        Retrieves uniquely indexed source URLs currently residing in the database 
        to ensure identical articles aren't re-embedded on subsequent runs.
        """
        try:
            current_data = self.vector_db.get(include=["metadatas"])
            if current_data and "metadatas" in current_data:
                return {
                    meta.get("source_url") 
                    for meta in current_data["metadatas"] 
                    if meta and "source_url" in meta
                }
        except Exception as e:
            logger.warning(f"Could not read existing metadata cache for deduplication: {e}")
        return set()

    def fetch_and_update(self):
        """
        Executes a single polling iteration across target media feeds, maps new events,
        and pushes incremental vector changes to Chroma DB.
        """
        logger.info("Polling active news syndication endpoints...")
        existing_links = self._get_existing_links()
        new_documents = []

        for source_name, feed_url in self.sources.items():
            try:
                # # Execute request within the authenticated session context
                # response = self.session.get(feed_url, timeout=15)
                
                # # If a feed hits a minor roadblock, log it clearly but keep processing other sources
                # if response.status_code == 403:
                #     logger.warning(f"[{source_name}] Cloudflare challenge met (403). Retrying with alternative format...")
                #     # Fallback modification for specific RSS endpoints
                #     response = self.session.get(feed_url, headers={'Accept': '*/*'}, timeout=15)
                # Impersonate a real Chrome 110 browser at the cryptographic level
                response = requests.get(feed_url, impersonate="chrome110", timeout=15)
                response.raise_for_status()
                
                # Parse the raw bytes directly to let feedparser handle compression formats (gzip/br)
                parsed_feed = feedparser.parse(response.content)
                entries_found = len(parsed_feed.entries)
                
                logger.info(f"Checking '{source_name}' - Active feed records found: {entries_found}")

                for entry in parsed_feed.entries:
                    link = getattr(entry, "link", None)
                    title = getattr(entry, "title", "")
                    description = getattr(entry, "description", "")
                    
                    if not link:
                        continue
                        
                    # Skip if article has already been vectorized on a previous execution pass
                    if link in existing_links:
                        continue

                    # Consolidate text content for semantic chunking
                    full_content = f"Title: {title}\nContent Summary: {description}"
                    
                    # Create normalized schema document object
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "source": source_name,
                            "source_url": link,
                            "timestamp": getattr(entry, "published", datetime.utcnow().isoformat()),
                            "ingested_at": datetime.utcnow().isoformat()
                        }
                    )
                    new_documents.append(doc)
                    existing_links.add(link)

            except Exception as feed_err:
                logger.error(f"Failed to query or parse feed stream for {source_name}: {feed_err}")

        # If new updates are found, pass them to your embedding engine
        if new_documents:
            logger.info(f"Injecting {len(new_documents)} brand-new verified items into ChromaDB...")
            self.vector_db.add_documents(new_documents)
            logger.info("Vector database updates successfully persisted onto the file system.")
        else:
            logger.info("Sync check complete. No new telemetry data detected. ChromaDB is up to date.")


def run_continuous_daemon(interval_seconds: int = 3600):
    """
    Infinite processing loop container that turns the script into a continuous background worker.
    """
    updater = RAGDatabaseUpdater()
    logger.info(f"Background worker active. Polling frequency set to every {interval_seconds} seconds.")
    
    try:
        while True:
            start_time = time.time()
            try:
                logger.info("Initiating RAG Database Update Cycle...")
                updater.fetch_and_update()
            except Exception as loop_err:
                logger.error(f"Unexpected error encountered during synchronization routine: {loop_err}")
                
            # Account for processing time overhead before executing the sleep window
            elapsed = time.time() - start_time
            sleep_duration = max(1, interval_seconds - elapsed)
            time.sleep(sleep_duration)
    except KeyboardInterrupt:
        logger.info("Database updater worker terminated cleanly by user command.")

if __name__ == "__main__":
    run_continuous_daemon(interval_seconds=3600)