# =============================================================================
# agent/tools.py — LangChain tools for the SHU RAG agent
#
# Two tools:
#   shu_knowledge_base  — searches the Chroma vectorstore (SHU-specific data)
#   web_search          — Tavily search for anything not in the KB
# =============================================================================
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List

from shu_rag.config.settings import FALLBACK_RESPONSE


def format_context(docs: List[Document]) -> str:
    """Convert a list of Documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_url", "shu.ac.uk")
        parts.append(f"[Document {i}]\n{doc.page_content}\nSource: {source}")
    return "\n\n---\n\n".join(parts)


def build_rag_tool(vectorstore, structured_llm, reranker):
    """
    Factory that returns the shu_knowledge_base tool with pipeline
    components injected via closure.
    """

    @tool
    def shu_knowledge_base(query: str) -> str:
        """
        Search Sheffield Hallam University's official knowledge base.
        Use this for questions about SHU courses, modules, fees, entry
        requirements, UCAS codes, accommodation, admissions, campus contacts,
        student support, and any other SHU-specific information.
        Always try this tool first before doing a web search.
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from shu_rag.prompts.system_prompt import SYSTEM_PROMPT
        from shu_rag.retrieval.query_intent import QueryIntent
        from shu_rag.retrieval.retriever import get_retriever_with_fallback
        from shu_rag.reranking.reranker import rerank
        from shu_rag.config.settings import DEFAULT_K

        try:
            query_intent = structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
        except Exception as e:
            print(f"[RAG Tool] query processor fallback: {e}")
            query_intent = QueryIntent(
                intents=["general"],
                k=DEFAULT_K,
                filters={},
                rewritten_query=query,
                is_greeting_or_chitchat=False,
            )

        docs = get_retriever_with_fallback(vectorstore, query_intent)
        reranked = rerank(
            query=query_intent.rewritten_query,
            docs=docs,
            reranker=reranker,
        )

        if not reranked:
            return "No relevant information found in the knowledge base."

        return format_context(reranked)

    return shu_knowledge_base


def build_web_search_tool():
    """
    Returns the Tavily web search tool.
    Requires TAVILY_API_KEY to be set in the environment.
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    return TavilySearchResults(
        max_results=5,
        description=(
            "Search the web for current information about Sheffield Hallam University "
            "or general study-related queries not found in the knowledge base. "
            "Use this when the knowledge base returns no useful results, or when the "
            "user asks about recent news, events, or general study advice."
        ),
    )