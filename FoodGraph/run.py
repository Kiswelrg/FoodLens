"""Run the FoodLens graph: a food photo path in, a grounded answer out.

    python run.py path/to/photo.jpg
    python run.py ../FoodVLM/data/food101_sample/*.jpg     # batch (traces 10+ runs)

Set LANGSMITH_API_KEY in the environment to record the runs in LangSmith
(project "FoodLens"); without it the graph still runs, just untraced.
"""
import argparse

import config
from graph import build_graph


def run_one(graph, image_path: str, verbose: bool = False) -> dict:
    state = graph.invoke({"image_path": image_path})
    if verbose:
        print(f"  vlm: {state.get('vlm_description', '')}")
        docs = state.get("retrieved_docs", [])
        if docs:
            print(f"  top hit: {docs[0]['score']:.3f}  {docs[0]['name']}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="FoodLens: image -> grounded answer.")
    ap.add_argument("images", nargs="+", help="path(s) to food photo(s)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the VLM description and top retrieval hit")
    args = ap.parse_args()

    traced = config.enable_langsmith()
    print(f"LangSmith tracing: {'ON (project=FoodLens)' if traced else 'off (no API key)'}\n")

    graph = build_graph()   # builds the Retriever once; reused across images
    for img in args.images:
        print(f"=== {img} ===")
        state = run_one(graph, img, verbose=args.verbose)
        print(state.get("answer", "(no answer)"), "\n")


if __name__ == "__main__":
    main()
