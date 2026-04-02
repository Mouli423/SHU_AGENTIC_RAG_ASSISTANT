# =============================================================================
# evaluate.py — RAGAS evaluation for the SHU RAG pipeline
#
# Usage:
#   python evaluate.py
#   python evaluate.py --sample 10 --output results/ragas_run1.csv
#
# Each question is evaluated independently — no chat history, no agent loop.
# Retrieval + generation run directly for clean, uncontaminated scores.
# =============================================================================

import argparse
import pandas as pd
from datasets import Dataset
from langchain_core.messages import SystemMessage, HumanMessage

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from shu_rag.ingestion.vectorstore import get_embeddings, load_vectorstore
from shu_rag.generation.llm import get_structured_llm, get_generator_llm
from shu_rag.reranking.reranker import load_reranker, rerank
from shu_rag.retrieval.retriever import get_retriever_with_fallback
from shu_rag.retrieval.query_intent import QueryIntent
from shu_rag.generation.generator import generate_answer
from shu_rag.prompts.system_prompt import SYSTEM_PROMPT
from shu_rag.config.settings import DEFAULT_K, RAGAS_SAMPLE_SIZE, RAGAS_OUTPUT_FILE


# =============================================================================
# Test set — ground truths verified against shu_all_chunks.json
# =============================================================================

TEST_SET = [
    {
        "question": "What is the UCAS code for BSc Computer Science at Sheffield Hallam University?",
        "ground_truth": "The UCAS code for BSc (Honours) Computer Science at Sheffield Hallam University is G400.",
    },
    {
        "question": "What are the UK tuition fees for the MSc Artificial Intelligence course?",
        "ground_truth": "The UK tuition fee for MSc Artificial Intelligence at Sheffield Hallam University is £10,940 for the course.",
    },
    {
        "question": "Does the MSc Artificial Intelligence course offer a placement year?",
        "ground_truth": (
            "The standard MSc Artificial Intelligence does not include a placement year. "
            "However, the MSc Artificial Intelligence (Work Experience) variant does include a placement."
        ),
    },
    {
        "question": "What are the compulsory modules in year 1 of BSc Computer Science?",
        "ground_truth": (
            "Year 1 compulsory modules for BSc Computer Science include "
            "Foundations in Computing 1, Foundations in Computing 2, and Study Skills for Computing Students."
        ),
    },
    {
        "question": "How can I apply for student accommodation at Sheffield Hallam University?",
        "ground_truth": (
            "You can apply for accommodation online at accom-online.shu.ac.uk. "
            "All full-time undergraduate students who firm SHU as their first choice are guaranteed "
            "an offer of halls accommodation in their first year."
        ),
    },
    {
        "question": "What are the entry requirements for BSc Data Science at Sheffield Hallam?",
        "ground_truth": "The entry requirements for BSc Data Science at Sheffield Hallam are 112–120 UCAS points.",
    },
    {
        "question": "What campus is the Computer Science degree taught at?",
        "ground_truth": "The BSc Computer Science degree at Sheffield Hallam University is taught at City Campus.",
    },
    {
        "question": "How do I contact the admissions team at Sheffield Hallam University?",
        "ground_truth": (
            "You can contact Sheffield Hallam University admissions by calling +44 (0)114 225 5555 "
            "or emailing enquiries@shu.ac.uk. Postgraduate admissions can be reached at pg.admissions@shu.ac.uk."
        ),
    },
    {
        "question": "What are the fees for BSc Nursing (Adult) at Sheffield Hallam?",
        "ground_truth": "The UK tuition fee for BSc Nursing (Adult) at Sheffield Hallam University is £7,155 per year.",
    },
    {
        "question": "Tell me about Sheffield Hallam University.",
        "ground_truth": (
            "Sheffield Hallam University is one of the UK's largest universities, founded in 1843 "
            "and located in Sheffield. It has approximately 34,000 students, a Gold TEF rating, "
            "and a 95% graduate employment rate."
        ),
    },
]


# =============================================================================
# Build RAGAS dataset — direct retrieval + generation, no agent loop
# =============================================================================

def run_single_question(question, vectorstore, structured_llm, generator_llm, reranker) -> dict:
    """
    Run one question through retrieval + generation independently.
    No chat history, no agent overhead — clean isolated evaluation.
    """
    # Step 1 — classify query
    try:
        query_intent = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
    except Exception as e:
        print(f"    [warn] query processor fallback: {e}")
        query_intent = QueryIntent(
            intents=["general"],
            k=DEFAULT_K,
            filters={},
            rewritten_query=question,
            is_greeting_or_chitchat=False,
        )

    # Step 2 — retrieve and rerank
    docs     = get_retriever_with_fallback(vectorstore, query_intent)
    reranked = rerank(query=query_intent.rewritten_query, docs=docs, reranker=reranker)
    contexts = [doc.page_content for doc in reranked]

    # Step 3 — generate answer
    answer = generate_answer(query_intent, reranked, generator_llm)

    return {"answer": answer, "contexts": contexts}


def build_ragas_dataset(test_set, vectorstore, structured_llm, generator_llm, reranker) -> Dataset:
    rows = []

    for i, item in enumerate(test_set):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i+1}/{len(test_set)}] {question[:70]}...")

        result = run_single_question(
            question, vectorstore, structured_llm, generator_llm, reranker
        )

        rows.append({
            "question":     question,
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": ground_truth,
        })

    return Dataset.from_list(rows)


# =============================================================================
# Run RAGAS metrics
# =============================================================================

def run_evaluation(dataset: Dataset, generator_llm, embeddings) -> pd.DataFrame:
    ragas_llm        = LangchainLLMWrapper(generator_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(ragas_llm),
            AnswerRelevancy(ragas_llm),
            ContextPrecision(ragas_llm),
            ContextRecall(ragas_llm),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return results.to_pandas()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on SHU RAG pipeline")
    parser.add_argument("--sample", type=int, default=RAGAS_SAMPLE_SIZE)
    parser.add_argument("--output", default=RAGAS_OUTPUT_FILE)
    return parser.parse_args()


def main():
    args     = parse_args()
    test_set = TEST_SET[:min(args.sample, len(TEST_SET))]

    print("=" * 60)
    print("Step 1 — Loading components")
    print("=" * 60)
    embeddings     = get_embeddings()
    vectorstore    = load_vectorstore(embeddings)
    structured_llm = get_structured_llm()
    generator_llm  = get_generator_llm()
    reranker       = load_reranker()

    print()
    print("=" * 60)
    print(f"Step 2 — Running {len(test_set)} questions")
    print("=" * 60)
    dataset = build_ragas_dataset(
        test_set, vectorstore, structured_llm, generator_llm, reranker
    )

    print()
    print("=" * 60)
    print("Step 3 — RAGAS metrics")
    print("=" * 60)
    df = run_evaluation(dataset, generator_llm, embeddings)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(df[cols].describe().round(3))

    df.to_csv(args.output, index=False)
    print(f"\nFull results saved → {args.output}")


if __name__ == "__main__":
    main()