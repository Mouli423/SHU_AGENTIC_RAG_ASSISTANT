# =============================================================================
# ingestion/vectorstore.py — embeddings and Chroma vector store
# =============================================================================

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from shu_rag.config.settings import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE,
    NORMALIZE_EMBEDDINGS, VECTORSTORE_DIR, COLLECTION_NAME, INGEST_BATCH_SIZE,
)


# =============================================================================
# Embeddings
# =============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={
            "batch_size": EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        },
    )


# =============================================================================
# Ingest documents in batches
# =============================================================================

def ingest_in_batches(
    docs,
    embeddings,
    persist_dir: str = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = INGEST_BATCH_SIZE,
) -> Chroma:
    """Ingest documents into Chroma in memory-safe batches."""
    vectorstore = None

    for i in tqdm(range(0, len(docs), batch_size), desc="Ingesting batches"):
        batch = docs[i : i + batch_size]
        ids   = [
            doc.metadata.get("chunk_id", f"id_{i + j}")
            for j, doc in enumerate(batch)
        ]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                ids=ids,
                persist_directory=persist_dir,
                collection_name=collection_name,
            )
        else:
            vectorstore.add_documents(batch)

    total = len(vectorstore.get()["ids"])
    print(f"Ingestion complete — {total} vectors stored")

    if total != len(docs):
        print(f"WARNING: expected {len(docs)}, got {total} — possible ID collision")
    else:
        print("All chunks ingested successfully")

    return vectorstore


# =============================================================================
# Load existing vectorstore
# =============================================================================

def load_vectorstore(
    embeddings=None,
    persist_dir: str = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load a previously persisted Chroma vectorstore."""
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
