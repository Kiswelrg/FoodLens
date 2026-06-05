"""FoodLens RAG as an MCP tool (stdio FastMCP server).

Exposes FoodRAG's three-stage retriever (dense + BM25 -> RRF -> rerank) as a
single MCP tool `recipe_search`, so any MCP client (a LangGraph agent, the MCP
Inspector, a desktop LLM app) can query the private 180k-recipe corpus. MCP's
value here is exactly the FoodLens thesis: a client may already have a VLM, but
it CANNOT reach this corpus with hybrid search + reranking -- that's the
load-bearing fact supplier. So the wrapped seam is RAG, not the VLM.

Run standalone:           python server.py
Inspect with MCP CLI:     mcp dev server.py
Use from an agent:        see agent_demo.py / README.md.

Knobs live as module constants below (NOT a `config.py`): this component imports
FoodRAG, whose modules do a bare `import config`, and a second top-level `config`
would collide (the same footgun FoodGraph/bridge.py documents). Keeping the few
knobs inline sidesteps it entirely.
"""
import importlib
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- knobs --------------------------------------------------------------------
SERVER_NAME = "foodlens-rag"
DEFAULT_MODE = "rerank"     # full pipeline: dense+sparse -> RRF -> rerank
DEFAULT_K = 5
VALID_MODES = ("dense", "sparse", "hybrid", "rerank")

# --- bridge to FoodRAG --------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FOODRAG_DIR = ROOT / "FoodRAG"


def _load_retriever_class():
    """Import FoodRAG's `retrieval.Retriever`.

    FoodRAG/retrieval.py does a bare `import config`; make that resolve to
    FoodRAG/config.py by putting FoodRAG's dir at the front of sys.path and
    evicting any cached `config`. Mirrors FoodGraph/bridge.py, but simpler: this
    server only ever loads FoodRAG (no FoodVLM), so there is no config *clash* to
    referee -- only this one resolution to pin down.
    """
    sys.modules.pop("config", None)
    sys.path.insert(0, str(FOODRAG_DIR))   # left on path: harmless, and lets any
    try:                                   # runtime lazy import inside FoodRAG resolve
        retrieval = importlib.import_module("retrieval")
    finally:
        pass
    return retrieval.Retriever


_RetrieverClass = _load_retriever_class()
_retriever = None   # lazy: the embedder/BM25/reranker are heavy to load


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _RetrieverClass()
    return _retriever


def _format_hit(hit: dict) -> dict:
    """Trim a raw retriever hit to a concise, LLM-groundable record.

    Keeps the human-readable `card` (the field we embed: name + description +
    ingredients + tags + calories) so the caller can answer FROM it, plus a few
    structured fields for filtering/ranking by the calling agent.
    """
    meta = hit.get("metadata", {})
    diets = [f for f in ("vegetarian", "vegan", "gluten_free",
                         "dairy_free", "low_carb") if meta.get(f)]
    return {
        "name": hit["name"],
        "score": round(float(hit["score"]), 4),
        "minutes": meta.get("minutes"),
        "calories": meta.get("nutrition_calories"),
        "n_ingredients": meta.get("n_ingredients"),
        "diets": diets,
        "card": hit["text"],
    }


# --- the MCP server -----------------------------------------------------------
mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def recipe_search(query: str, mode: str = DEFAULT_MODE, k: int = DEFAULT_K) -> list[dict]:
    """Search the FoodLens recipe corpus and return grounded recipe cards.

    Use this to look up real recipes, exact nutrition (calories), cooking time,
    diet suitability (vegetarian/vegan/gluten-free/dairy-free/low-carb), and the
    ingredient/step card for a dish -- facts a model should not guess. Good
    queries mix exact tokens (ingredient names, "gluten-free", a calorie target)
    with natural-language intent ("a comforting cold-weather stew").

    Args:
        query: Natural-language or keyword description of the dish/constraints.
        mode: Retrieval mode. "rerank" (default) runs the full pipeline
            dense+BM25 -> RRF fusion -> cross-encoder rerank. Others: "hybrid"
            (fusion, no rerank), "dense" (semantic only), "sparse" (BM25 only).
        k: Number of recipes to return (default 5).

    Returns:
        A list of recipe records, best first, each with: name, score (higher is
        a better match), minutes, calories, n_ingredients, diets (list of
        matched diet flags), and card (the full human-readable recipe text to
        ground an answer on).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    hits = _get_retriever().search(query, mode=mode, final_k=k)
    return [_format_hit(h) for h in hits]


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
