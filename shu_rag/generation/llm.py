# =============================================================================
# generation/llm.py — LLM client setup
# =============================================================================

from langchain_aws import ChatBedrockConverse
from shu_rag.retrieval.query_intent import QueryIntent
from shu_rag.config.settings import (
    AWS_REGION, QUERY_MODEL_ID, GENERATOR_MODEL_ID, GENERATOR_TEMPERATURE,
)


def get_query_llm() -> ChatBedrockConverse:
    """Return the LLM used for structured query processing."""
    return ChatBedrockConverse(
        model_id=QUERY_MODEL_ID,
        region_name=AWS_REGION,
    )


def get_structured_llm():
    """Return the query LLM bound to the QueryIntent output schema."""
    return get_query_llm().with_structured_output(QueryIntent)


def get_generator_llm() -> ChatBedrockConverse:
    """Return the LLM used for answer generation."""
    return ChatBedrockConverse(
        model_id=GENERATOR_MODEL_ID,
        region_name=AWS_REGION,
        temperature=GENERATOR_TEMPERATURE,
    )
