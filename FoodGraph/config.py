"""FoodGraph config -- the LangGraph capstone that wires VLM -> RAG -> summary.

FoodGraph is the third sibling component. It owns NO models or data of its own;
it imports the perception seam from FoodVLM and the retrieval/summary seams from
FoodRAG (via bridge.py) and orchestrates them as a StateGraph.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- Retrieval / summary knobs (the graph's own orchestration choices) --------
RETRIEVAL_MODE = "rerank"     # full dense+sparse -> RRF -> rerank pipeline
SUMMARY_BACKEND = "deepseek"  # grounded answer LLM (see FoodRAG/summarize.py)

# Conditional-edge threshold. The eyeball test showed that when the VLM misreads
# a dish, the top rerank score collapses (~0.19) instead of returning confident
# garbage. We use the top score as a confidence proxy: below this, the graph
# branches to a clarify node ("ask the user") instead of summarizing a likely
# wrong recipe. bge-reranker-v2-m3 scores are unbounded logits; ~0.3 cleanly
# separated the strong hits (>0.9) from the french-onion miss (0.188).
CONFIDENCE_THRESHOLD = 0.3


def enable_langsmith() -> bool:
    """Turn on LangSmith tracing IFF an API key is present in the environment.

    Set LANGSMITH_API_KEY (or legacy LANGCHAIN_API_KEY) to trace runs; otherwise
    the graph still runs, just untraced. Returns whether tracing was enabled.
    """
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "FoodLens")
    return True
