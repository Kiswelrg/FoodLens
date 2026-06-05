"""FoodVLM perception node: turn a food photo into a RAG-usable query string.

This is the front half of the VLM->RAG seam, but it lives in its OWN component
(FoodVLM), separate from FoodRAG. `describe_image(path)` returns a short,
retrieval-friendly description -- NOT a paragraph. In the Week 3 LangGraph build
this becomes the `vlm_node`, and its output becomes the `query` string handed to
FoodRAG's retriever; nothing in FoodRAG changes.

Why a terse, structured description instead of a caption: BM25 wants exact
tokens (ingredient names, dish names) and the dense embedder wants salient
attributes. A flowery caption ("a delightful bowl of warmth...") dilutes both.
So we prompt the VLM for: dish name + key visible ingredients + a couple of
salient attributes + a rough portion estimate in grams.

Model: InternVL3-8B (4-bit) via mlx-vlm on Apple Silicon. The model is heavy to
load (~seconds), so it is lazy-loaded once and cached at module scope.

The optional `--search` flag runs the resulting queries through FoodRAG to
eyeball retrieval quality. It does so in a SUBPROCESS rooted at the FoodRAG dir,
so the two components' identically-named `config` modules never collide -- which
is also how the two stay decoupled until the LangGraph build wires them.
"""
import argparse

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

import config

# The instruction that shapes the VLM output into a retrieval query. We ask for
# a single structured line so the result is dense with the exact tokens BM25
# keys on and the attributes the dense embedder rewards -- and explicitly forbid
# the prose a "describe this image" prompt would otherwise produce.
PROMPT = (
    "You are labeling a food photo to search a recipe database. "
    "Look at the dish and output ONE concise line (no full sentences, no preamble) "
    "containing, in order:\n"
    "1. the most likely dish name,\n"
    "2. the key visible ingredients,\n"
    "3. one or two salient attributes (e.g. spicy, creamy, grilled, vegetarian),\n"
    "4. a rough total portion estimate in grams.\n"
    "Format exactly like:\n"
    "<dish name>; ingredients: <a, b, c>; attributes: <x, y>; approx <N> g\n"
    "Only describe what is visibly present. Do not guess hidden ingredients."
)

# Module-level cache: (model, processor, model_config). Loaded on first call.
_VLM: "tuple | None" = None


def _get_vlm():
    global _VLM
    if _VLM is None:
        model, processor = load(config.VLM_MODEL)
        model_config = load_config(config.VLM_MODEL)
        _VLM = (model, processor, model_config)
    return _VLM


def describe_image(path: str, max_tokens: "int | None" = None,
                   verbose: bool = False) -> str:
    """Photo -> compact, retrieval-ready query string.

    Deterministic (temperature 0) so the same dish yields a stable query, which
    keeps the downstream retrieval benchmark reproducible. Internal newlines are
    flattened so the result is a single clean query line.
    """
    model, processor, model_config = _get_vlm()
    formatted = apply_chat_template(processor, model_config, PROMPT, num_images=1)
    result = generate(
        model,
        processor,
        formatted,
        image=[path],
        max_tokens=max_tokens or config.VLM_MAX_TOKENS,
        temperature=0.0,
        verbose=verbose,
    )
    return " ".join(result.text.split())


# --- optional cross-component eyeball test (VLM query -> FoodRAG retrieval) ----
_RAG_RUNNER = """
import sys
from retrieval import Retriever
r = Retriever()
for line in sys.stdin:
    q = line.rstrip("\\n")
    if not q:
        continue
    print("\\nQUERY: " + q)
    for h in r.search(q, mode="rerank"):
        print("    %.3f  %s" % (h["score"], h["name"]))
"""


def _run_through_rag(queries: list[str]) -> None:
    """Feed queries into FoodRAG in a subprocess rooted at the FoodRAG dir."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-c", _RAG_RUNNER],
        cwd=str(config.FOODRAG_DIR),
        input="\n".join(queries),
        text=True,
        check=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Describe food image(s) as RAG queries.")
    ap.add_argument("images", nargs="+", help="path(s) to food photo(s)")
    ap.add_argument("--search", action="store_true",
                    help="also run each description through FoodRAG (subprocess)")
    ap.add_argument("--verbose", action="store_true", help="stream VLM generation")
    args = ap.parse_args()

    queries = []
    for img in args.images:
        query = describe_image(img, verbose=args.verbose)
        print(f"# {img}\n{query}\n")
        queries.append(query)

    if args.search:
        print("=== FoodRAG retrieval for each description ===")
        _run_through_rag(queries)
