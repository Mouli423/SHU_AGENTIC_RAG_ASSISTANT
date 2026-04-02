# =============================================================================
# generation/generator.py — context formatting and answer generation
# =============================================================================

from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from shu_rag.retrieval.query_intent import QueryIntent
from shu_rag.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT
from shu_rag.config.settings import FALLBACK_RESPONSE


# =============================================================================
# Format retrieved docs into a context string
# =============================================================================

def format_context(docs: List[Document]) -> str:
    """Convert a list of Documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_url", "shu.ac.uk")
        parts.append(f"[Document {i}]\n{doc.page_content}\nSource: {source}")
    return "\n\n---\n\n".join(parts)


# =============================================================================
# Answer generation
# =============================================================================

def generate_answer(
    query_intent: QueryIntent,
    docs: List[Document],
    generator_llm,
) -> str:
    """
    Generate a grounded answer from retrieved documents.

    Returns the fallback response string if no docs are provided.
    """
    if not docs:
        return FALLBACK_RESPONSE

    context = format_context(docs)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {query_intent.rewritten_query}"
        )),
    ]

    response = generator_llm.invoke(messages)
    return StrOutputParser().invoke(response)
