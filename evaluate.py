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
from ragas.metrics import (
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
        "ground_truth": (
            "The UCAS code for BSc (Honours) Computer Science at Sheffield Hallam University is G400. "
            "The course is offered full-time at City Campus with entry requirements of 112-120 UCAS points. "
            "The UK tuition fee is £9,790 per year and the international fee is £18,000 per year. "
            "A placement year is available. Highlights include gaining skills to write software for diverse "
            "industries, developing understanding of AI, robotics and machine learning, and benefiting from "
            "connections with employers such as IBM, Intel, Next and HSBC."
        ),
    },
    {
        "question": "What are the UK tuition fees for the MSc Artificial Intelligence course?",
        "ground_truth": (
            "The UK tuition fee for MSc Artificial Intelligence at Sheffield Hallam University is £10,940 "
            "for the course. The international fee is £18,600 for the course. The course is full-time, "
            "located at City Campus. It does not include a placement year. Highlights include studying "
            "modern AI practices, theories, techniques and ethical considerations, developing problem-solving "
            "skills by applying AI methods, and cultivating a substantial portfolio of commercial-quality work."
        ),
    },
    {
        "question": "Does the MSc Artificial Intelligence course offer a placement year?",
        "ground_truth": (
            "The standard MSc Artificial Intelligence at Sheffield Hallam University does not include a "
            "placement year (Placement: No). However, the MSc Artificial Intelligence (Work Experience) "
            "variant does include a placement year (Placement: Yes). The Work Experience variant costs "
            "£12,440 for the course (UK) and £20,100 (international), compared to £10,940 (UK) and "
            "£18,600 (international) for the standard course. Both are full-time and located at City Campus."
        ),
    },
    {
        "question": "What are the compulsory modules in BSc Computer Science at Sheffield Hallam?",
        "ground_truth": (
            "Compulsory modules for BSc (Honours) Computer Science at Sheffield Hallam University include: "
            "Databases and the Web (20 credits, Coursework 100%) — covers creating dynamic websites underpinned "
            "by databases; Mathematics for Computer Science (20 credits, Coursework 100%) — introduces discrete "
            "mathematics and numerical skills; Programming for Computer Science (20 credits, Coursework 100%) — "
            "provides an introduction to computer programming and object-oriented concepts. "
            "All modules are assessed 100% by coursework."
        ),
    },
    {
        "question": "How can I apply for student accommodation at Sheffield Hallam University?",
        "ground_truth": (
            "Sheffield Hallam University offers 11 halls of residence properties, all within walking distance "
            "of both campuses. All full-time undergraduate students who firm SHU as their first choice are "
            "guaranteed an offer of accommodation in halls for their first year. Weekly prices range from "
            "£81 to £185 per week (2025/26). All bills are included — gas, electricity, water, Wi-Fi, and "
            "contents insurance. There is 24-hour security and a residential support team. "
            "Applications open for September 2026 entry. You can apply online at accom-online.shu.ac.uk."
        ),
    },
    {
        "question": "What are the entry requirements for BSc Data Science at Sheffield Hallam?",
        "ground_truth": (
            "The entry requirements for BSc (Honours) Data Science at Sheffield Hallam University are "
            "112-120 UCAS points. The UCAS code is BB35. The course is full-time, 3 or 4 years duration, "
            "located at City Campus. The UK tuition fee is £9,790 per year and international fee is "
            "£18,000 per year. A placement year is available. Highlights include designing innovative "
            "systems using data science, harnessing AI and data mining to extract insights from large "
            "datasets, and gaining skills in programming and cloud computing."
        ),
    },
    {
        "question": "What campus is the Computer Science degree taught at?",
        "ground_truth": (
            "The BSc (Honours) Computer Science degree at Sheffield Hallam University is taught at "
            "City Campus, located at Howard Street, Sheffield, S1 1WB. The course is full-time with "
            "a UCAS code of G400, entry requirements of 112-120 UCAS points, UK fee of £9,790 per year, "
            "and a placement year option."
        ),
    },
    {
        "question": "How do I contact the admissions team at Sheffield Hallam University?",
        "ground_truth": (
            "To contact Sheffield Hallam University: "
            "General enquiries — main switchboard: +44 (0)114 225 5555, email: enquiries@shu.ac.uk, "
            "address: City Campus, Howard Street, Sheffield, S1 1WB. "
            "Student Services (Hallam Help) — phone: 0114 225 4321 (Mon–Fri 9am–5pm), "
            "email: hallamhelp@shu.ac.uk, online: hallamhelp.shu.ac.uk. "
            "Undergraduate admissions: 0114 225 5555. "
            "Postgraduate admissions: pg.admissions@shu.ac.uk. "
            "International admissions: international.admissions@shu.ac.uk."
        ),
    },
    {
        "question": "What are the fees for BSc Nursing (Adult) at Sheffield Hallam?",
        "ground_truth": (
            "The UK tuition fee for BSc (Honours) Nursing (Adult) at Sheffield Hallam University is "
            "£7,155 per year. The course is part-time, located at City Campus. Highlights include "
            "gaining the knowledge and skills to provide compassionate nursing care, developing a strong "
            "professional identity, and entering highly skilled employment. The course is in the "
            "nursing-and-midwifery subject area."
        ),
    },
    {
        "question": "Tell me about Sheffield Hallam University.",
        "ground_truth": (
            "Sheffield Hallam University (SHU) is one of the UK's largest universities, located in "
            "Sheffield, South Yorkshire. Key facts: Founded in 1843 as the Sheffield School of Design, "
            "gained university status in 1992. Approximately 34,000 students including 5,000+ international "
            "students, and around 4,000 staff. TEF Rating: Gold (2023 Teaching Excellence Framework). "
            "QS Stars: 5-Star Excellence Rating. Ranked in the Top 10 UK student cities (Uni Compare 2025). "
            "95% of graduates are in work or further study within 15 months (Graduate Outcomes Survey 2022/23). "
            "UCAS institution code: S21. Two main campuses: City Campus at Howard Street, Sheffield S1 1WB, "
            "and Collegiate Campus at Psalter Lane, Sheffield S11 8UZ. "
            "General enquiries: +44 (0)114 225 5555 or enquiries@shu.ac.uk."
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
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
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