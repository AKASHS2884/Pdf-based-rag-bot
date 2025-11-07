#FAQ - With DB Integration(PSQL)

#main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import json
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
from io import StringIO
import logging
from datetime import datetime
import re
import os
import asyncio
import requests

# Database imports
import asyncpg
from contextlib import asynccontextmanager
import uuid

# Retrieval stack
# pip install: fastapi uvicorn sentence-transformers faiss-cpu transformers torch pandas numpy requests asyncpg python-dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------------------
# Database Configuration
# -------------------------------------------------
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Cvhs@12345"),
    "database": os.getenv("DB_NAME", "postgres")
}

# Global database connection pool
db_pool: Optional[asyncpg.Pool] = None


# -------------------------------------------------
# Database Functions
# -------------------------------------------------
async def init_db_pool():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            database=DATABASE_CONFIG["database"],
            min_size=2,
            max_size=10
        )
        logger.info("Database pool initialized successfully")

        # Create table if it doesn't exist
        await create_sample_table()

    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        db_pool = None


async def close_db_pool():
    """Close database connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")


async def create_sample_table():
    """Create the sample table with proper structure, fixing any existing issues"""
    if not db_pool:
        logger.warning("Database pool not available")
        return

    try:
        async with db_pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval("""
                                               SELECT EXISTS (SELECT
                                                              FROM information_schema.tables
                                                              WHERE table_schema = 'public'
                                                                AND table_name = 'sample');
                                               """)

            if table_exists:
                # Check if the table has the correct structure
                columns = await conn.fetch("""
                                           SELECT column_name, data_type, is_nullable, column_default
                                           FROM information_schema.columns
                                           WHERE table_name = 'sample'
                                             AND table_schema = 'public'
                                           ORDER BY ordinal_position;
                                           """)

                column_info = {row['column_name']: row for row in columns}
                logger.info(f"Existing table structure: {list(column_info.keys())}")

                # Check if ID column has proper SERIAL setup
                id_column = column_info.get('id')
                needs_recreation = False

                if not id_column:
                    needs_recreation = True
                    logger.warning("ID column missing")
                elif 'nextval' not in str(id_column['column_default'] or ''):
                    needs_recreation = True
                    logger.warning("ID column doesn't have proper SERIAL default")

                # Check for other required columns
                required_columns = ['user_query', 'response', 'conversation_id', 'source', 'confidence', 'created_at']
                missing_columns = [col for col in required_columns if col not in column_info]

                if missing_columns:
                    logger.warning(f"Missing columns: {missing_columns}")
                    needs_recreation = True

                if needs_recreation:
                    logger.info("Recreating table with proper structure...")

                    # Create backup
                    await conn.execute("CREATE TABLE IF NOT EXISTS sample_backup AS SELECT * FROM sample;")
                    logger.info("Created backup table")

                    # Drop and recreate
                    await conn.execute("DROP TABLE sample CASCADE;")
                    logger.info("Dropped old table")

                    # Create new table
                    create_table_query = """
                                         CREATE TABLE sample \
                                         ( \
                                             id              SERIAL PRIMARY KEY, \
                                             user_query      TEXT NOT NULL, \
                                             response        TEXT NOT NULL, \
                                             created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP, \
                                             conversation_id VARCHAR(255), \
                                             source          VARCHAR(50), \
                                             confidence      FLOAT
                                         ); \
                                         """
                    await conn.execute(create_table_query)
                    logger.info("Created new table with proper structure")

                    # Migrate data back if backup exists
                    backup_count = await conn.fetchval("SELECT COUNT(*) FROM sample_backup;")
                    if backup_count > 0:
                        migrate_query = """
                                        INSERT INTO sample (user_query, response, created_at, conversation_id, source, confidence)
                                        SELECT COALESCE(user_query, 'Unknown query')                               as user_query, \
                                               COALESCE(response, 'Unknown response')                              as response, \
                                               COALESCE(created_at, CURRENT_TIMESTAMP)                             as created_at, \
                                               COALESCE(conversation_id, \
                                                        'legacy_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP) \
                                                        ::text)                                                    as conversation_id, \
                                               COALESCE(source, 'unknown')                                         as source, \
                                               COALESCE(confidence, 0.0)                                           as confidence
                                        FROM sample_backup; \
                                        """
                        await conn.execute(migrate_query)
                        migrated_count = await conn.fetchval("SELECT COUNT(*) FROM sample;")
                        logger.info(f"Migrated {migrated_count} rows from backup")

                else:
                    logger.info("Table structure is correct, no recreation needed")

            else:
                # Create new table
                logger.info("Creating new sample table...")
                create_table_query = """
                                     CREATE TABLE sample \
                                     ( \
                                         id              SERIAL PRIMARY KEY, \
                                         user_query      TEXT NOT NULL, \
                                         response        TEXT NOT NULL, \
                                         created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP, \
                                         conversation_id VARCHAR(255), \
                                         source          VARCHAR(50), \
                                         confidence      FLOAT
                                     ); \
                                     """
                await conn.execute(create_table_query)
                logger.info("Created new sample table")

            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_conversation_id ON sample(conversation_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_created_at ON sample(created_at);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_sample_source ON sample(source);")
            logger.info("Created/verified indexes")

        logger.info("Sample table setup completed successfully")

    except Exception as e:
        logger.error(f"Failed to create/update sample table: {e}")
        raise


async def store_conversation(user_query: str, response: str, conversation_id: str = None,
                             source: str = "faq", confidence: float = 0.0):
    """Store conversation in database with proper error handling"""
    if not db_pool:
        logger.warning("Database pool not available, skipping storage")
        return None

    # Generate conversation_id if not provided
    if not conversation_id:
        conversation_id = f"conv_{int(time.time())}"

    insert_query = """
                   INSERT INTO sample (user_query, response, conversation_id, source, confidence, created_at)
                   VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP) RETURNING id; \
                   """

    try:
        async with db_pool.acquire() as conn:
            row_id = await conn.fetchval(
                insert_query,
                user_query,
                response,
                conversation_id,
                source,
                float(confidence) if confidence is not None else 0.0
            )
        logger.info(f"Stored conversation with ID: {row_id}")
        return row_id
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")
        # Try to get more details about the error
        try:
            async with db_pool.acquire() as conn:
                # Check if table exists and has correct structure
                table_info = await conn.fetch("""
                                              SELECT column_name, data_type, is_nullable, column_default
                                              FROM information_schema.columns
                                              WHERE table_name = 'sample'
                                                AND table_schema = 'public'
                                              ORDER BY ordinal_position;
                                              """)
                logger.error(f"Current table structure: {[dict(row) for row in table_info]}")
        except Exception as check_error:
            logger.error(f"Could not check table structure: {check_error}")
        return None


async def get_conversation_history(conversation_id: str = None, limit: int = 50):
    """Retrieve conversation history"""
    if not db_pool:
        return []

    if conversation_id:
        query = """
                SELECT id, user_query, response, created_at, source, confidence
                FROM sample
                WHERE conversation_id = $1
                ORDER BY created_at DESC
                    LIMIT $2; \
                """
        params = [conversation_id, limit]
    else:
        query = """
                SELECT id, user_query, response, created_at, source, confidence, conversation_id
                FROM sample
                ORDER BY created_at DESC
                    LIMIT $1; \
                """
        params = [limit]

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve conversation history: {e}")
        return []


async def get_db_stats():
    """Get database statistics"""
    if not db_pool:
        return {"error": "Database not available"}

    stats_query = """
                  SELECT COUNT(*)                                      as total_conversations, \
                         COUNT(DISTINCT conversation_id)               as unique_conversations, \
                         COUNT(CASE WHEN source = 'faq' THEN 1 END)    as faq_responses, \
                         COUNT(CASE WHEN source = 'system' THEN 1 END) as system_responses, \
                         AVG(confidence)                               as avg_confidence, \
                         MAX(created_at)                               as last_conversation
                  FROM sample; \
                  """

    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(stats_query)
        return dict(row) if row else {}
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}


# -------------------------------------------------
# App and logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db_pool()
    yield
    # Shutdown
    await close_db_pool()


app = FastAPI(
    title="iPhone 17 RAG Chatbot with PostgreSQL",
    version="7.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Simple static bearer token auth (for your API)
# -------------------------------------------------
STATIC_TOKEN = os.getenv("STATIC_TOKEN", "Cvhs@12345")


class StaticTokenBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> dict:
        creds: HTTPAuthorizationCredentials = await super().__call__(request)
        if not creds or creds.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        if creds.credentials != STATIC_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")
        return {"authenticated": True}


require_token = StaticTokenBearer()


# -------------------------------------------------
# Models
# -------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    source: str  # "faq" or "system"
    confidence: float
    conversation_id: str
    retrieved_chunks: Optional[List[Dict]] = None
    db_stored: bool = False


class ConversationHistoryRequest(BaseModel):
    conversation_id: Optional[str] = None
    limit: int = 50


# -------------------------------------------------
# Global state (RAG)
# -------------------------------------------------
faq_rows: List[Dict[str, str]] = []
chunk_store: List[Dict[str, str]] = []

bi_encoder: Optional[SentenceTransformer] = None
ce_reranker: Optional[CrossEncoder] = None
faiss_index: Optional[faiss.IndexFlatIP] = None
chunk_embs: Optional[np.ndarray] = None
embedding_dim: Optional[int] = None

# -------------------------------------------------
# OpenRouter (online LLM) configuration
# -------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
# Fallback to provided demo key ONLY if env not set (not recommended for production)
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "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"  # demo
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "iPhone 17 RAG Chatbot")


def call_openrouter_chat(messages: List[Dict[str, str]], max_tokens: int = 220,
                         temperature: float = 0.85, top_p: float = 0.92) -> str:
    """
    Call OpenRouter chat completions API with OpenAI-compatible schema.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_X_TITLE,
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # Optional: penalties to reduce repetition
        "presence_penalty": 0.0,
        "frequency_penalty": 0.2,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        logger.error(f"OpenRouter call failed: {e}")
        raise


# -------------------------------------------------
# Chunking + preprocessing
# -------------------------------------------------
PRODUCT_ALIASES = {
    "iphone 17": {"iphone 17", "iphone17", "ip17", "17"},
    "iphone": {"iphone", "apple phone", "apple iphone"}
}


def detect_product_entities(text: str) -> Set[str]:
    t = re.sub(r'\s+', ' ', text.lower()).strip()
    hits: Set[str] = set()
    for canonical, aliases in PRODUCT_ALIASES.items():
        if any(alias in t for alias in aliases):
            hits.add(canonical)
    return hits


def product_match_bonus(chunk: dict, query_entities: Set[str]) -> float:
    if not query_entities:
        return 0.0
    text = f"{chunk.get('question', '')} {chunk.get('text', '')}".lower()
    bonus = 0.0
    for canonical in query_entities:
        if canonical in text:
            bonus += 0.05
    return min(bonus, 0.15)


def sentence_split(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    sents = re.split(r'(?<=[\.\!\?])\s+', text) if text else []
    return [s for s in sents if s]


def to_word_chunks(sentences: List[str], chunk_size_words: int = 120, overlap_words: int = 30) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(' '.join(buf).strip())
            if overlap_words > 0:
                words = ' '.join(buf).split()
                overlap = words[-overlap_words:] if len(words) > overlap_words else words
                buf = [' '.join(overlap)]
                buf_len = len(overlap)
            else:
                buf = []
                buf_len = 0

    for s in sentences:
        w = s.split()
        if buf_len + len(w) <= chunk_size_words or not buf:
            buf.append(s)
            buf_len += len(w)
        else:
            flush()
            buf.append(s)
            buf_len = len(w)
    if buf:
        chunks.append(' '.join(buf).strip())
    return chunks


def preprocess(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


# -------------------------------------------------
# Retrieval + online Generation
# -------------------------------------------------
def expand_query(q: str) -> str:
    ql = q.strip().lower()
    if len(ql.split()) < 3:
        topics = {
            "battery": "battery life, charging (USB‑C, MagSafe), and power efficiency",
            "camera": "camera system, portraits, low‑light, and 4K video",
            "display": "Super Retina XDR brightness, HDR, and readability",
            "performance": "chip performance, multitasking, and connectivity",
        }
        for k, v in topics.items():
            if k in ql:
                return f"iPhone 17 {k} overview focusing on {v}"
        return "iPhone 17 overview covering camera, display, battery, performance, and durability"
    return q


def clean_sentences(text: str) -> List[str]:
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    out = []
    for s in sents:
        if len(s) < 30:
            continue
        out.append(s)
    return out


def compose_context(chunks: List[Dict[str, str]], max_bullets: int = 8) -> str:
    seen = set()
    bullets = []
    for c in chunks:
        for s in clean_sentences(c.get("text", "")):
            if s not in seen:
                seen.add(s)
                bullets.append(f"- {s}")
            if len(bullets) >= max_bullets:
                break
        if len(bullets) >= max_bullets:
            break
    return "\n".join(bullets) if bullets else "- No additional details specified in the FAQ."


class RAGChatbot:
    def __init__(self, k: int = 5, topN: int = 24):
        self.top_k = k
        self.topN = topN
        self.intent_keywords = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"],
            "help": ["help", "assistance", "support"],
        }

    def extract_intent(self, message: str) -> str:
        m = message.lower()
        for intent, kws in self.intent_keywords.items():
            if any(k in m for k in kws):
                return intent
        return "query"

    def retrieve(self, query: str) -> Tuple[List[Dict[str, str]], List[float]]:
        global chunk_store, bi_encoder, ce_reranker, chunk_embs, faiss_index
        if faiss_index is None or chunk_embs is None or bi_encoder is None or not chunk_store:
            return [], []

        q_emb = bi_encoder.encode(
            [expand_query(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )[0].astype(np.float32)

        k = min(max(self.topN, self.top_k), len(chunk_store))
        sims, idxs = faiss_index.search(q_emb.reshape(1, -1), k)
        sims = sims[0];
        idxs = idxs[0]

        # Light entity-aware bonus (no exclusions)
        query_entities = detect_product_entities(query + " iphone 17")
        if idxs.size > 0 and query_entities:
            boosted = []
            for s, i in zip(sims, idxs):
                boosted.append(s + product_match_bonus(chunk_store[i], query_entities))
            sims = np.array(boosted, dtype=np.float32)

        if idxs.size == 0:
            return [], []

        # Rerank
        top_texts = [chunk_store[i]["text"] for i in idxs]
        pairs = [[query, t] for t in top_texts]
        try:
            rerank_scores = ce_reranker.predict(pairs).astype(np.float32) if ce_reranker else sims
        except Exception:
            rerank_scores = sims

        # Normalize for confidence
        if rerank_scores.size > 1:
            mn, mx = float(np.min(rerank_scores)), float(np.max(rerank_scores))
            norm = (rerank_scores - mn) / (mx - mn + 1e-9)
        else:
            norm = rerank_scores

        order = np.argsort(-rerank_scores)[: self.top_k]
        final_idx = idxs[order]
        final_scores = norm[order].tolist()
        return [chunk_store[i] for i in final_idx], final_scores

    def build_messages(self, query: str, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        bullets = compose_context(chunks, max_bullets=8)
        qx = expand_query(query)
        system_prompt = (
            "You are a helpful Apple product expert.\n"
            "Answer ONLY about iPhone 17 (base model). Do NOT include links, CTAs, or contact details.\n"
            "Be enthusiastic and customer‑friendly, and write ONE cohesive paragraph, 3–5 sentences, no lists.\n"
            "Use ONLY the context; if a detail isn't in the context, say it isn't specified."
        )
        user_content = f"Context:\n{bullets}\n\nQuestion:\n{qx}\n\nAnswer:"
        # OpenRouter supports OpenAI-compatible messages with optional system role
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def limit_sentences(self, text: str, min_s: int = 3, max_s: int = 5) -> str:
        parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        parts = [p for p in parts if p]
        if not parts:
            return text.strip()
        clipped = " ".join(parts[:max_s]).strip()
        return clipped if len(parts) >= min_s else text.strip()

    async def generate_response(self, message: str, conversation_id: str = None) -> Dict:
        low = message.strip().lower()
        if "what phone are we discussing" in low or "which phone are we discussing" in low:
            response_data = {"response": "We are discussing iPhone 17.", "source": "system", "confidence": 1.0}
        else:
            intent = self.extract_intent(message)
            if intent == "greeting":
                response_data = {
                    "response": "Hello! Ask anything about iPhone 17; answers are grounded in your FAQ chunks.",
                    "source": "system", "confidence": 1.0}
            elif intent == "goodbye":
                response_data = {"response": "Goodbye! Come back anytime with more iPhone 17 questions.",
                                 "source": "system", "confidence": 1.0}
            else:
                chunks, scores = self.retrieve(message)
                conf = float(scores[0]) if scores else 0.0

                if chunks:
                    messages = self.build_messages(message, chunks)
                    try:
                        text = await asyncio.to_thread(call_openrouter_chat, messages)
                        text = self.limit_sentences(text, min_s=3, max_s=5)
                        response_data = {
                            "response": text,
                            "source": "faq",
                            "confidence": conf,
                            "retrieved_chunks": [
                                {"question": c.get("question", ""), "text": c.get("text", "")[:200] + "..."} for c in
                                chunks[:3]]
                        }
                    except Exception as e:
                        logger.error(f"OpenRouter generation error: {e}")
                        text = self.extractive_summary(chunks=chunks, query=message)
                        response_data = {"response": text, "source": "faq", "confidence": conf}
                else:
                    response_data = {
                        "response": "Sorry, no relevant iPhone 17 FAQ chunks were found. Please upload FAQs or rephrase the question.",
                        "source": "system", "confidence": 0.0}

        # Store in database
        db_stored = False
        try:
            if conversation_id:
                row_id = await store_conversation(
                    user_query=message,
                    response=response_data["response"],
                    conversation_id=conversation_id,
                    source=response_data["source"],
                    confidence=response_data["confidence"]
                )
                db_stored = row_id is not None
        except Exception as e:
            logger.error(f"Failed to store conversation in database: {e}")

        response_data["db_stored"] = db_stored
        return response_data

    def extractive_summary(self, chunks: List[Dict[str, str]], query: str, max_chars: int = 900) -> str:
        entities = detect_product_entities(query + " iphone 17")
        product_terms = set()
        for e in entities:
            product_terms.update(PRODUCT_ALIASES.get(e, {e}))
        product_terms.update(entities)
        sents: list[str] = []
        for c in chunks:
            text = c.get("text", "")
            for s in re.split(r'(?<=[\.\!\?])\s+', text):
                if not s.strip():
                    continue
                s_lower = s.lower()
                if not product_terms or any(term in s_lower for term in product_terms):
                    sents.append(s.strip())
        seen = set();
        picked = []
        for s in sents:
            if s not in seen:
                picked.append(s);
                seen.add(s)
            if len(" ".join(picked)) > max_chars:
                break
        return " ".join(picked) if picked else "No relevant details available in the current FAQ chunks."


chatbot = RAGChatbot(k=5, topN=24)


# -------------------------------------------------
# Initialization
# -------------------------------------------------
def init_models_if_needed():
    global bi_encoder, ce_reranker
    if bi_encoder is None:
        bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded bi‑encoder: sentence-transformers/all-MiniLM-L6-v2")
    if ce_reranker is None:
        try:
            ce_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Loaded cross‑encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"Cross‑encoder unavailable: {e}")
            ce_reranker = None


# -------------------------------------------------
# Data loading + FAISS indexing
# -------------------------------------------------
def load_faq_data(content: str, filename: str,
                  chunk_size_words: int = 120,
                  overlap_words: int = 30) -> bool:
    global faq_rows, chunk_store, chunk_embs, faiss_index, embedding_dim
    try:
        init_models_if_needed()
        if filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content))
        elif filename.endswith('.json'):
            data = json.loads(content);
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format")

        df.columns = df.columns.str.strip().str.lower()
        if 'question' not in df.columns or 'answer' not in df.columns:
            q_candidates = [c for c in df.columns if any(term in c for term in ['question', 'q', 'query'])]
            a_candidates = [c for c in df.columns if any(term in c for term in ['answer', 'a', 'response'])]
            if q_candidates and a_candidates:
                df = df.rename(columns={q_candidates[0]: 'question', a_candidates[0]: 'answer'})
            else:
                raise ValueError(f"FAQ file must contain 'question' and 'answer' columns. Found: {list(df.columns)}")

        df = df[['question', 'answer']].dropna()
        df = df[df['question'].str.strip() != ''];
        df = df[df['answer'].str.strip() != '']
        if df.empty:
            raise ValueError("No valid question-answer pairs found")
        faq_rows = df.to_dict('records')

        # Build chunks (no special exclusions)
        chunk_store = []
        chunk_id = 0
        for idx, row in enumerate(faq_rows):
            q = str(row['question']).strip()
            a = str(row['answer']).strip()
            if not a:
                continue
            sents = sentence_split(a)
            chunks = to_word_chunks(sents, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
            for ch in chunks:
                if ch.strip():
                    tags = list(detect_product_entities(q + " " + ch))
                    if "iphone 17" not in tags:
                        tags.append("iphone 17")
                    chunk_store.append({
                        "id": f"chunk_{chunk_id}",
                        "faq_idx": idx,
                        "question": q,
                        "text": ch,
                        "tags": tags
                    })
                    chunk_id += 1

        if not chunk_store:
            raise ValueError("No chunks produced from the uploaded FAQ.")

        texts = [c["text"] for c in chunk_store]
        embs = bi_encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)
        chunk_embs = embs;
        embedding_dim = embs.shape[1]

        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_index.add(chunk_embs)

        logger.info(f"Loaded {len(faq_rows)} FAQ rows, {len(chunk_store)} chunks, FAISS indexed: {faiss_index.ntotal}")
        return True
    except Exception as e:
        logger.error(f"Error loading FAQ data: {e}")
        faq_rows.clear();
        chunk_store.clear()
        return False


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.get("/auth/verify")
async def auth_verify(claims: dict = Depends(require_token)):
    return {"authenticated": True, "message": "Token is valid"}


@app.post("/upload-faq")
async def upload_faq(file: UploadFile = File(...), claims: dict = Depends(require_token)):
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        if load_faq_data(content_str, file.filename or "uploaded.csv"):
            return JSONResponse(status_code=200, content={
                "message": f"FAQ file '{file.filename}' uploaded successfully. Chunked into {len(chunk_store)} and indexed in FAISS."})
        else:
            raise HTTPException(status_code=400, detail="Failed to process FAQ file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, claims: dict = Depends(require_token)):
    try:
        conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        response_data = await chatbot.generate_response(request.message, conversation_id)
        return ChatResponse(
            response=response_data["response"],
            source=response_data["source"],
            confidence=response_data["confidence"],
            conversation_id=conversation_id,
            retrieved_chunks=response_data.get("retrieved_chunks"),
            db_stored=response_data.get("db_stored", False)
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")


@app.post("/conversation-history")
async def get_conversation_history_endpoint(request: ConversationHistoryRequest, claims: dict = Depends(require_token)):
    """Get conversation history from database"""
    try:
        history = await get_conversation_history(request.conversation_id, request.limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@app.get("/db-stats")
async def get_db_stats_endpoint(claims: dict = Depends(require_token)):
    """Get database statistics"""
    try:
        stats = await get_db_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")


@app.get("/health")
async def health_check():
    db_status = "connected" if db_pool else "disconnected"
    return {
        "status": "healthy",
        "faq_loaded": len(chunk_store) > 0,
        "models_loaded": {
            "bi_encoder": bi_encoder is not None,
            "cross_encoder": ce_reranker is not None,
            "online_llm": bool(OPENROUTER_API_KEY)
        },
        "database_status": db_status,
        "online_model": OPENROUTER_MODEL,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/faq-stats")
async def get_faq_stats():
    return {
        "total_faqs": len(faq_rows),
        "total_chunks": len(chunk_store),
        "faiss_index_ntotal": int(faiss_index.ntotal) if faiss_index is not None else 0,
        "embedding_dim": embedding_dim,
        "models": {
            "bi_encoder": "sentence-transformers/all-MiniLM-L6-v2" if bi_encoder else None,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2" if ce_reranker else None,
            "online_llm": OPENROUTER_MODEL
        },
        "sample_chunk_preview": chunk_store[0]["text"][:200] + "..." if chunk_store else "",
        "sample_tags": chunk_store[0]["tags"] if chunk_store else []
    }


if __name__ == "__main__":
    import uvicorn

    try:
        init_models_if_needed()
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
    uvicorn.run(app, host=os.getenv("BACKEND_HOST", "0.0.0.0"), port=int(os.getenv("BACKEND_PORT", 8000)),
                log_level="info")
