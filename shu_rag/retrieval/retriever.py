# =============================================================================
# retrieval/retriever.py — vector store retrieval with three-level fallback
# =============================================================================

from shu_rag.retrieval.query_intent import QueryIntent
from shu_rag.retrieval.filters import build_chroma_filter, CHUNK_TYPE_MAP


def get_retriever_with_fallback(vectorstore, query_intent: QueryIntent):
    """
    Retrieve documents from Chroma with a three-level fallback strategy:

      1. Primary   — full metadata filter (intents + user-specified fields)
      2. Fallback  — chunk_type filter only (drop user-specified fields)
      3. Last resort — pure vector search (no filter at all)

    Uses MMR search for module_detail to promote diversity,
    similarity search for everything else.
    """
    intents     = query_intent.intents
    k           = query_intent.k
    filters     = query_intent.filters or {}
    search_type = "mmr" if "module_detail" in intents else "similarity"

    # ── Level 1: full filter ──────────────────────────────────────────────────
    where = build_chroma_filter(intents, filters)
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "filter": where},
    )
    docs = retriever.invoke(query_intent.rewritten_query)

    if docs:
        return docs

    # ── Level 2: chunk_type filter only ───────────────────────────────────────
    print("[Retriever] filtered search returned nothing — falling back to chunk_type only")

    chunk_types = []
    for intent in intents:
        chunk_types.extend(CHUNK_TYPE_MAP.get(intent, []))

    fallback_filter = (
        {"chunk_type": chunk_types[0]}
        if len(chunk_types) == 1
        else {"chunk_type": {"$in": chunk_types}}
    )
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "filter": fallback_filter},
    )
    docs = retriever.invoke(query_intent.rewritten_query)

    if docs:
        return docs

    # ── Level 3: pure vector search ───────────────────────────────────────────
    print("[Retriever] chunk_type fallback also failed — falling back to pure vector search")
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return retriever.invoke(query_intent.rewritten_query)
