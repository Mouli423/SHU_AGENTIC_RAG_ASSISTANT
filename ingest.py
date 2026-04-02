# =============================================================================
# ingest.py — run this ONCE to build the Chroma vectorstore from merged data
#
# Usage:
#   python ingest.py
#   python ingest.py --merged data/shu_all_chunks.json
# =============================================================================

import argparse

from shu_rag.ingestion.loader import load_documents
from shu_rag.ingestion.vectorstore import get_embeddings, ingest_in_batches
from shu_rag.config.settings import MERGED_FILE, VECTORSTORE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest SHU chunks into Chroma")
    parser.add_argument("--merged",      default=MERGED_FILE,     help="Path to merged chunks JSON")
    parser.add_argument("--vectorstore", default=VECTORSTORE_DIR, help="Chroma persist directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Step 1 — Loading documents")
    print("=" * 60)
    docs = load_documents(args.merged)

    print()
    print("=" * 60)
    print("Step 2 — Loading embedding model")
    print("=" * 60)
    embeddings = get_embeddings()

    print()
    print("=" * 60)
    print("Step 3 — Ingesting into Chroma")
    print("=" * 60)
    ingest_in_batches(docs, embeddings, persist_dir=args.vectorstore)

    print()
    print("Ingestion complete. You can now run: python main.py")


if __name__ == "__main__":
    main()