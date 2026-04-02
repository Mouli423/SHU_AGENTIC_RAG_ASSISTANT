# =============================================================================
# pipeline/pipeline.py — end-to-end RAG pipeline orchestrator
# =============================================================================

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from shu_rag.retrieval.query_intent import QueryIntent
from shu_rag.retrieval.retriever import get_retriever_with_fallback
from shu_rag.reranking.reranker import rerank
from shu_rag.generation.generator import generate_answer
from shu_rag.prompts.system_prompt import SYSTEM_PROMPT
from shu_rag.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT
from shu_rag.config.settings import DEFAULT_K


class SHUPipeline:

    def __init__(self, vectorstore, structured_llm, generator_llm, reranker):
        self.vectorstore    = vectorstore
        self.structured_llm = structured_llm
        self.generator_llm  = generator_llm
        self.reranker       = reranker

    def process_query(self, query: str) -> QueryIntent:
        try:
            return self.structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ])
        except Exception as e:
            print(f"[QueryProcessor] fallback triggered: {e}")
            return QueryIntent(
                intents=["general"],
                k=DEFAULT_K,
                filters={},
                rewritten_query=query,
                is_greeting_or_chitchat=False,
            )

    def _retrieve_and_rerank(self, query_intent: QueryIntent):
        docs = get_retriever_with_fallback(self.vectorstore, query_intent)
        return rerank(
            query=query_intent.rewritten_query,
            docs=docs,
            reranker=self.reranker,
        )

    def _handle_chitchat(self, query: str) -> str:
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = self.generator_llm.invoke(messages)
        return StrOutputParser().invoke(response)

    def ask(self, query: str) -> str:
        query_intent = self.process_query(query)

        if query_intent.is_greeting_or_chitchat:
            return self._handle_chitchat(query)

        reranked_docs = self._retrieve_and_rerank(query_intent)
        return generate_answer(query_intent, reranked_docs, self.generator_llm)