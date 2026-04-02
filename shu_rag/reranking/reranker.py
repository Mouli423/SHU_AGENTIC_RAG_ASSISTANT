# =============================================================================
# reranking/reranker.py — CrossEncoder reranker
# =============================================================================

from typing import List, Optional
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from shu_rag.config.settings import RERANKER_MODEL, RERANKER_DEVICE


# =============================================================================
# Load reranker (singleton — call once at startup)
# =============================================================================

def load_reranker() -> CrossEncoder:
    """Load and return the CrossEncoder reranker model."""
    return CrossEncoder(model_name=RERANKER_MODEL, device=RERANKER_DEVICE)


# =============================================================================
# Rerank function
# =============================================================================

def rerank(
    query: str,
    docs: List[Document],
    reranker: CrossEncoder,
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Score each (query, doc) pair with the CrossEncoder and return docs
    sorted by descending relevance score.

    Args:
        query:    The user's rewritten query string.
        docs:     Candidate documents from the retriever.
        reranker: Loaded CrossEncoder instance.
        top_n:    How many to return. Defaults to all docs (let the
                  caller decide how many it needs).

    Returns:
        Documents sorted by rerank score, truncated to top_n.
    """
    if not docs:
        return []

    if top_n is None:
        top_n = len(docs)

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.metadata["_rerank_score"] = round(float(score), 4)

    reranked = sorted(docs, key=lambda d: d.metadata["_rerank_score"], reverse=True)
    return reranked[:top_n]
