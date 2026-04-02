# =============================================================================
# retrieval/filters.py — chunk type mapping and Chroma filter builder
# =============================================================================

from typing import List
from shu_rag.config.settings import DEFAULT_ENTRY_YEAR

# =============================================================================
# Chunk type mappings
# =============================================================================

GENERAL_CHUNK_TYPES = [
    "overview", "campus", "admissions", "fees",
    "accommodation", "international", "support",
    "student_life", "contacts",
]

CHUNK_TYPE_MAP = {
    "course_summary": ["course_summary"],
    "module_detail":  ["module_detail"],
    "general":        GENERAL_CHUNK_TYPES,
}

ALLOWED_FILTER_KEYS = {
    "course", "subject", "course_level", "entry_year",
    "degree_type", "study_mode", "placement", "location",
    "module", "year", "module_section", "assessment",
    "subcategory", "target_audience",
}

# ── course_level normalisation ─────────────────────────────────────────────────
# The LLM generates "postgraduate" but the data stores "postgraduate_taught".
COURSE_LEVEL_MAP = {
    "postgraduate":        "postgraduate_taught",
    "postgraduate_taught": "postgraduate_taught",
    "undergraduate":       "undergraduate",
    "foundation":          "foundation",
}

# ── Keys to always strip from filters ─────────────────────────────────────────
# entry_year  — chunks span 2025/2026/missing; filtering drops valid results
# placement   — work experience variants store Placement as empty string not
#               "Yes", so filtering on placement misses them entirely. Let the
#               LLM see both standard and work experience chunks and compare.
STRIP_FILTER_KEYS = {"entry_year", "placement"}


# =============================================================================
# Build the Chroma $where filter dict
# =============================================================================

def build_chroma_filter(intents: List[str], filters: dict) -> dict:
    """
    Construct a Chroma-compatible $where filter from intents and metadata filters.

    Keys in STRIP_FILTER_KEYS are always excluded — they either vary too much
    across chunks (entry_year) or are stored inconsistently (placement).
    """
    # Resolve chunk types from intents
    chunk_types = []
    for intent in intents:
        chunk_types.extend(CHUNK_TYPE_MAP.get(intent, []))

    # Build chunk_type condition
    if len(chunk_types) == 1:
        conditions = [{"chunk_type": {"$eq": chunk_types[0]}}]
    else:
        conditions = [{"chunk_type": {"$in": chunk_types}}]

    # Add remaining metadata filters
    for key, value in filters.items():
        if value is None or key in STRIP_FILTER_KEYS:
            continue

        # Normalise course_level to match stored values
        if key == "course_level":
            value = COURSE_LEVEL_MAP.get(value.lower(), value)

        conditions.append({key: {"$eq": value}})

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]