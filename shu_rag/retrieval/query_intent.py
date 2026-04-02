# =============================================================================
# retrieval/query_intent.py — Pydantic schema for structured query output
# =============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any

from shu_rag.config.settings import MIN_K, MAX_K


class QueryIntent(BaseModel):
    intents: List[str] = Field(
        description=(
            "One or more retrieval intents. Each must be one of: "
            "'course_summary', 'module_detail', 'general'. "
            "Include multiple if the query spans more than one area."
        )
    )
    k: int = Field(
        description="Number of documents to retrieve based on query complexity."
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metadata filters to apply during vector store retrieval. "
            "Only include fields explicitly mentioned or clearly implied. "
            "Empty dict if no filters apply."
        ),
    )
    rewritten_query: str = Field(
        description=(
            "Clean, standalone, expanded query for vector search. "
            "Expand abbreviations, fix typos, resolve pronouns, add SHU context."
        )
    )
    is_greeting_or_chitchat: bool = Field(
        default=False,
        description=(
            "True if the query is a greeting, chitchat, or completely off-topic. "
            "No retrieval needed."
        ),
    )

    @field_validator("intents")
    @classmethod
    def validate_intents(cls, v, values):
        if values.data.get("is_greeting_or_chitchat"):
            return v or ["general"]

        valid   = {"course_summary", "module_detail", "general"}
        invalid = [i for i in v if i not in valid]

        if invalid:
            raise ValueError(f"Invalid intents: {invalid}")
        if not v:
            raise ValueError("intents must not be empty")

        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v):
        return max(MIN_K, min(MAX_K, v))
