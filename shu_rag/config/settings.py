# =============================================================================
# config/settings.py — central configuration for SHU RAG assistant
# =============================================================================

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR               = "data"
MERGED_FILE            = f"{DATA_DIR}/shu_all_chunks.json"

# ── Vectorstore ───────────────────────────────────────────────────────────────
VECTORSTORE_DIR        = "./shu_vectorstore"
COLLECTION_NAME        = "shu_chunks"
INGEST_BATCH_SIZE      = 500

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE       = "cpu"
EMBEDDING_BATCH_SIZE   = 64
NORMALIZE_EMBEDDINGS   = True

# ── LLM (AWS Bedrock) ─────────────────────────────────────────────────────────
AWS_REGION             = "us-east-1"
QUERY_MODEL_ID         = "amazon.nova-lite-v1:0"
GENERATOR_MODEL_ID     = "amazon.nova-lite-v1:0"
GENERATOR_TEMPERATURE  = 0.3

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL         = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE        = "cpu"

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_K              = 12
MIN_K                  = 3
MAX_K                  = 40
DEFAULT_ENTRY_YEAR     = "2026"

# ── Fallback response ─────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "I couldn't find relevant information for your query. "
    "Please visit shu.ac.uk or contact the admissions team at "
    "enquiries@shu.ac.uk or call +44 (0)114 225 5555."
)

# ── LangSmith tracing ─────────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2   = True                    # set False to disable
LANGCHAIN_PROJECT      = "shu-rag-assistant"     # project name in LangSmith

# ── RAGAS evaluation ──────────────────────────────────────────────────────────
RAGAS_SAMPLE_SIZE      = 20    # number of QA pairs to evaluate
RAGAS_OUTPUT_FILE      = "ragas_results.csv"

# ── Tavily web search ─────────────────────────────────────────────────────────
TAVILY_MAX_RESULTS     = 5    # number of web results returned per search

# ── Agent ─────────────────────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS   = 5    # safety limit on agent reasoning steps