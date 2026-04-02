# =============================================================================
# ingestion/loader.py — document loading and metadata extraction
# =============================================================================

import json
import re
from pathlib import Path
from langchain_community.document_loaders import JSONLoader

from shu_rag.config.settings import MERGED_FILE

# =============================================================================
# Regex pattern sets — one per chunk type
# =============================================================================

COURSE_PATTERNS = {
    "degree_type":         r"Degree Type:\s*(.+)",
    "study_mode":          r"Mode:\s*([^|\n]+)",
    "ucas_code":           r"UCAS Code:\s*([A-Z0-9]{4})",
    "entry_requirements":  r"Entry Requirements:\s*(.+)",
    "uk_fee":              r"UK Fee:\s*(£[\d,]+[^|\n]*)",
    "int_fee":             r"Int Fee:\s*(£[\d,]+[^|\n]*)",
    "placement":           r"Placement:\s*(Yes|No)",
    "location":            r"Location:\s*(.+)",
}

MODULE_PATTERNS = {
    "degree_type":    r"Course:.+?\((.+?)\)(?:\s|$)",
    "module_section": r"Type:\s*([^|\n]+)\s*modules",
    "module_credits": r"Credits:\s*(\d+)",
    "assessment":     r"Assessment:\s*(.+)",
}

GENERAL_CHUNK_TYPES = {
    "overview", "campus", "admissions", "fees",
    "accommodation", "international", "support",
    "student_life", "contacts",
}


# =============================================================================
# Metadata extraction function — called by JSONLoader for every chunk
# =============================================================================

def metadata_func(sample: dict, meta: dict) -> dict:
    """
    Extracts and promotes all metadata fields from a chunk.

    `sample` = the full chunk dict (one element from chunks[])
    `meta`   = JSONLoader's built-in metadata (source file path, seq_num)

    Returns a flat metadata dict that LangChain attaches to the Document.
    """
    text       = sample.get("text", "")
    chunk_type = sample.get("chunk_type", "")
    meta_sub   = sample.get("metadata", {})

    # ── Step 1: promote structured chunk-level fields ─────────────────────────
    base = {
        "chunk_id":        sample.get("chunk_id", ""),
        "chunk_type":      chunk_type,
        "source_url":      sample.get("source_url", ""),
        "course":          sample.get("course", ""),
        "subject":         sample.get("subject", ""),
        "module":          sample.get("module", ""),
        "year":            sample.get("year", ""),
        "category":        meta_sub.get("category", ""),
        "course_level":    meta_sub.get("course_level", ""),
        "entry_year":      meta_sub.get("entry_year", ""),
        "confidence":      meta_sub.get("confidence", "high"),
        "subcategory":     sample.get("subcategory", meta_sub.get("subcategory", "")),
        "target_audience": meta_sub.get("target_audience", "all"),
    }

    # ── Step 2: regex-extract text fields by chunk type ───────────────────────
    def extract(pattern, src=text):
        if pattern is None:
            return ""
        m = re.search(pattern, src, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    if chunk_type == "course_summary":
        base.update({
            "degree_type":        extract(COURSE_PATTERNS["degree_type"]),
            "study_mode":         extract(COURSE_PATTERNS["study_mode"]).strip(" |"),
            "ucas_code":          extract(COURSE_PATTERNS["ucas_code"]),
            "entry_requirements": extract(COURSE_PATTERNS["entry_requirements"]),
            "uk_fee":             extract(COURSE_PATTERNS["uk_fee"]).strip(),
            "int_fee":            extract(COURSE_PATTERNS["int_fee"]).strip(),
            "placement":          extract(COURSE_PATTERNS["placement"]),
            "location":           extract(COURSE_PATTERNS["location"]),
        })

    elif chunk_type == "module_detail":
        base.update({
            "degree_type":    extract(MODULE_PATTERNS["degree_type"]),
            "module_section": extract(MODULE_PATTERNS["module_section"]),
            "module_credits": extract(MODULE_PATTERNS["module_credits"]),
            "assessment":     extract(MODULE_PATTERNS["assessment"]),
        })

    elif chunk_type in GENERAL_CHUNK_TYPES:
        base.update({
            "uk_fee":  extract(r"UK Fee:\s*(£[\d,]+[^|\n]*)"),
            "int_fee": extract(r"Int Fee:\s*(£[\d,]+[^|\n]*)"),
        })

    return base


# =============================================================================
# Data merging — combine course chunks + supplementary KB
# =============================================================================

# def merge_chunks(
#     chunks_file: str = CHUNKS_FILE,
#     supplement_file: str = SUPPLEMENT_FILE,
#     output_file: str = MERGED_FILE,
# ) -> str:
#     """Merge course chunks and supplementary KB into a single file."""
#     courses_data = json.load(open(chunks_file, encoding="utf-8"))
#     supplement   = json.load(open(supplement_file, encoding="utf-8"))

#     course_chunks = courses_data["chunks"]
#     supp_chunks   = supplement["chunks"]
#     all_chunks    = course_chunks + supp_chunks

#     merged = {"total_chunks": len(all_chunks), "chunks": all_chunks}

#     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(merged, f, ensure_ascii=False, indent=2)

#     print(f"Course chunks:     {len(course_chunks)}")
#     print(f"Supplement chunks: {len(supp_chunks)}")
#     print(f"Total:             {len(all_chunks)}")
#     print(f"Saved → {output_file}")

#     return output_file


# =============================================================================
# Document loading
# =============================================================================

def load_documents(merged_file: str = MERGED_FILE):
    """Load all chunks from the merged JSON file as LangChain Documents."""
    loader = JSONLoader(
        file_path=merged_file,
        jq_schema=".chunks[]",
        content_key="text",
        metadata_func=metadata_func,
        text_content=False,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs
