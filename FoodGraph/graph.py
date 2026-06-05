"""The FoodLens capstone: image path in -> grounded answer out, as a StateGraph.

    vlm_node(image) -> rag_node(query) -> summarize_node(query, docs) -> END
                                  \\
                                   -> clarify_node  (if VLM confidence is low)

Each node is a thin wrapper over a seam imported through bridge.py:
  * vlm_node       -> FoodVLM.describe_image   (perception)
  * rag_node       -> FoodRAG.Retriever.search (facts)
  * summarize_node -> FoodRAG.summarize        (grounded LLM answer)

The conditional edge after rag_node uses the top rerank score as a VLM-confidence
proxy (see config.CONFIDENCE_THRESHOLD): a confident retrieval flows to the
summary; a collapsed score (the honest failure mode found in the eyeball test)
routes to clarify_node instead of confidently summarizing the wrong recipe.

The Retriever is heavy (loads embeddings + BM25 + lazy reranker), so we build it
once and close over it in the node rather than per invocation.
"""
from typing import TypedDict

from langgraph.graph import END, StateGraph

import config
from bridge import Retriever, describe_image, summarize


class FoodState(TypedDict, total=False):
    image_path: str          # input
    vlm_description: str      # produced by vlm_node, used as the retrieval query
    retrieved_docs: list      # produced by rag_node
    top_score: float          # confidence proxy for the conditional edge
    answer: str              # produced by summarize_node / clarify_node


def build_graph(retriever=None, summary_backend: str | None = None):
    """Compile the StateGraph. Pass a shared `retriever` to avoid reloading the
    indices; otherwise one is constructed on first build."""
    retriever = retriever or Retriever()
    backend = summary_backend or config.SUMMARY_BACKEND

    def vlm_node(state: FoodState) -> FoodState:
        return {"vlm_description": describe_image(state["image_path"])}

    def rag_node(state: FoodState) -> FoodState:
        hits = retriever.search(state["vlm_description"], mode=config.RETRIEVAL_MODE)
        top = hits[0]["score"] if hits else 0.0
        return {"retrieved_docs": hits, "top_score": top}

    def summarize_node(state: FoodState) -> FoodState:
        ans = summarize(state["vlm_description"], state["retrieved_docs"],
                        backend=backend)
        return {"answer": ans}

    def clarify_node(state: FoodState) -> FoodState:
        # Low confidence: don't summarize a likely-wrong recipe. Surface the
        # VLM's read and ask the user to confirm -- the planned graceful branch.
        return {"answer": (
            "I'm not confident I recognized this dish "
            f"(top match scored {state.get('top_score', 0.0):.2f}). "
            f"My best read was: \"{state['vlm_description']}\". "
            "Could you tell me what the dish is, or send a clearer photo?"
        )}

    def route_on_confidence(state: FoodState) -> str:
        return ("summarize_node"
                if state.get("top_score", 0.0) >= config.CONFIDENCE_THRESHOLD
                else "clarify_node")

    g = StateGraph(FoodState)
    g.add_node("vlm_node", vlm_node)
    g.add_node("rag_node", rag_node)
    g.add_node("summarize_node", summarize_node)
    g.add_node("clarify_node", clarify_node)

    g.set_entry_point("vlm_node")
    g.add_edge("vlm_node", "rag_node")
    g.add_conditional_edges("rag_node", route_on_confidence,
                            {"summarize_node": "summarize_node",
                             "clarify_node": "clarify_node"})
    g.add_edge("summarize_node", END)
    g.add_edge("clarify_node", END)
    return g.compile()
