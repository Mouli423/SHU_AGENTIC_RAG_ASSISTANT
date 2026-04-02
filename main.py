# =============================================================================
# main.py — conversational CLI for the SHU RAG agent
# Type your question and press Enter to get an answer.
# Type 'q' or 'Q' to exit.
# =============================================================================

from tracing import setup_tracing
setup_tracing()   # must be called before any LangChain imports

from shu_rag.ingestion.vectorstore import get_embeddings, load_vectorstore
from shu_rag.generation.llm import get_structured_llm, get_generator_llm
from shu_rag.reranking.reranker import load_reranker
from shu_rag.agent.agent import SHUAgent


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         Sheffield Hallam University — AI Assistant           ║
║         Type your question and press Enter                   ║
║         Type  q  or  Q  to exit                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def build_agent() -> SHUAgent:
    print("Starting up...")
    print("  Loading embeddings...")
    embeddings = get_embeddings()

    print("  Loading vectorstore...")
    vectorstore = load_vectorstore(embeddings)

    print("  Loading LLMs...")
    structured_llm = get_structured_llm()
    generator_llm  = get_generator_llm()

    print("  Loading reranker...")
    reranker = load_reranker()

    print("  Ready.\n")
    return SHUAgent(vectorstore, structured_llm, generator_llm, reranker)


def main():
    agent = build_agent()
    print(BANNER)
    print(f"Session ID: {agent.session_id}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "q":
            print("Goodbye!")
            break

        print()
        answer = agent.ask(query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()