
# =============================================================================
# tracing.py — LangSmith tracing setup
#
# Call setup_tracing() once at the top of main.py before anything else.
#
# Required env vars (add to .env or export in shell):
#   LANGCHAIN_API_KEY   — your LangSmith API key
#
# Optional overrides:
#   LANGCHAIN_PROJECT   — defaults to settings.LANGCHAIN_PROJECT
# =============================================================================

import os
from shu_rag.config.settings import LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
from dotenv import load_dotenv
load_dotenv()

def setup_tracing():
    """Configure LangSmith tracing via environment variables."""

    if not LANGCHAIN_TRACING_V2:
        print("[Tracing] LangSmith tracing is disabled.")
        return

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[Tracing] WARNING: LANGSMITH_API_KEY not set — tracing will not work.")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"]     = api_key
    os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

    print(f"[Tracing] LangSmith enabled → project: {os.environ['LANGCHAIN_PROJECT']}")