"""Citable retrieval benchmark: random / dense / sparse / hybrid / hybrid+rerank.

Two complementary scorings, both over LLM-generated, leak-checked queries
(see eval_set.py) — NOT the recipe's own text, so absolute numbers are honest:

1. Known-item retrieval (cheap, deterministic): each query was written for one
   specific recipe; that recipe is the single gold doc. We report MRR@10 and
   Hit@{1,5,10}, with a bootstrap 95% CI on MRR so the lift is defensible.

2. RAGAS (semantic, LLM-judged): context precision (does the ranking put
   genuinely-relevant recipes first? — judged by DeepSeek against a reference
   answer) and context recall (did we retrieve the gold recipe card?
   non-LLM string match). This credits a method for surfacing *other* relevant
   recipes that known-item MRR would wrongly count as misses.

Usage:
    python evaluate.py                 # full run (known-item + RAGAS)
    python evaluate.py --no-ragas      # fast: known-item metrics only
    python evaluate.py --ragas-n 30    # cap the (LLM-judged) RAGAS subset
"""
import argparse
import asyncio
import random

import numpy as np

import config
from eval_set import load as load_eval_set
from retrieval import Retriever

EVAL_K = 10              # retrieval depth for MRR / Hit@k
RAGAS_K = config.FINAL_TOPK   # context size RAGAS scores (the product setting)
MODES = ("random", "dense", "sparse", "hybrid", "rerank")
BOOTSTRAP_B = 1000


# ----------------------------- retrieval -----------------------------
def retrieve(r: Retriever, query: str, mode: str, k: int, rng: random.Random) -> list[dict]:
    """Unified retrieval incl. a `random` floor baseline not in Retriever."""
    if mode == "random":
        idxs = rng.sample(range(len(r.docs)), k)
        return [{**r.docs[i], "score": 0.0} for i in idxs]
    return r.search(query, mode=mode, final_k=k)


def _rank_of_gold(hits: list[dict], gold_id: int) -> "int | None":
    for rank, h in enumerate(hits):
        if h["id"] == gold_id:
            return rank
    return None


# ----------------------------- known-item -----------------------------
def known_item(r: Retriever, eval_set: list[dict]) -> dict:
    rng = random.Random(0)
    results = {}
    for mode in MODES:
        rrs = []
        hit1 = hit5 = hit10 = 0
        for item in eval_set:
            hits = retrieve(r, item["query"], mode, EVAL_K, rng)
            rank = _rank_of_gold(hits, item["gold_id"])
            if rank is None:
                rrs.append(0.0)
            else:
                rrs.append(1.0 / (rank + 1))
                hit1 += rank == 0
                hit5 += rank < 5
                hit10 += rank < 10
        rrs = np.array(rrs)
        lo, hi = _bootstrap_ci(rrs)
        n = len(eval_set)
        results[mode] = {
            "MRR@10": float(rrs.mean()),
            "MRR_ci": (lo, hi),
            "Hit@1": hit1 / n,
            "Hit@5": hit5 / n,
            "Hit@10": hit10 / n,
        }
    return results


def _bootstrap_ci(values: np.ndarray, b: int = BOOTSTRAP_B,
                  seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------- RAGAS -------------------------------
def ragas_eval(r: Retriever, eval_set: list[dict], modes: tuple[str, ...],
               n: int, concurrency: int = 8) -> dict:
    # Imported lazily so --no-ragas needs neither ragas nor a DeepSeek key.
    from ragas_compat import (SingleTurnSample, make_context_precision,
                              make_context_recall)

    subset = eval_set[:n]
    precision = make_context_precision()
    recall = make_context_recall()
    rng = random.Random(1)

    async def score_one(metric, sample) -> float:
        try:
            return float(await metric.single_turn_ascore(sample))
        except Exception:  # noqa: BLE001 — a flaky judge call shouldn't kill the run
            return float("nan")

    async def run_mode(mode: str) -> dict:
        sem = asyncio.Semaphore(concurrency)
        samples = []
        for item in subset:
            hits = retrieve(r, item["query"], mode, RAGAS_K, rng)
            samples.append(SingleTurnSample(
                user_input=item["query"],
                retrieved_contexts=[h["text"] for h in hits],
                reference=item["reference"],
                reference_contexts=[item["gold_text"]],
            ))

        async def guarded(metric, s):
            async with sem:
                return await score_one(metric, s)

        prec = await asyncio.gather(*(guarded(precision, s) for s in samples))
        rec = await asyncio.gather(*(guarded(recall, s) for s in samples))
        return {
            "context_precision": float(np.nanmean(prec)),
            "context_recall": float(np.nanmean(rec)),
        }

    async def run_all():
        out = {}
        for mode in modes:
            print(f"  RAGAS scoring mode={mode} on {len(subset)} queries ...")
            out[mode] = await run_mode(mode)
        return out

    return asyncio.run(run_all())


# ------------------------------- output ------------------------------
def print_known_item(results: dict, n: int) -> None:
    print(f"\nKnown-item retrieval — {n} LLM-generated, leak-checked queries\n")
    print(f"{'mode':<10}{'MRR@10':>9}{'  (95% CI)':>18}{'Hit@1':>9}{'Hit@5':>9}{'Hit@10':>9}")
    for mode, m in results.items():
        ci = f"[{m['MRR_ci'][0]:.3f},{m['MRR_ci'][1]:.3f}]"
        print(f"{mode:<10}{m['MRR@10']:>9.3f}{ci:>18}"
              f"{m['Hit@1']:>9.3f}{m['Hit@5']:>9.3f}{m['Hit@10']:>9.3f}")


def print_ragas(results: dict, n: int) -> None:
    print(f"\nRAGAS (top-{RAGAS_K} contexts) — DeepSeek judge, {n} queries\n")
    print(f"{'mode':<10}{'context_precision':>20}{'context_recall':>17}")
    for mode, m in results.items():
        print(f"{mode:<10}{m['context_precision']:>20.3f}{m['context_recall']:>17.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ragas", action="store_true", help="skip LLM-judged RAGAS")
    ap.add_argument("--ragas-n", type=int, default=60,
                    help="number of queries in the (LLM-judged) RAGAS subset")
    args = ap.parse_args()

    eval_set = load_eval_set()
    r = Retriever()

    ki = known_item(r, eval_set)
    print_known_item(ki, len(eval_set))

    if not args.no_ragas:
        ragas_modes = ("random", "dense", "sparse", "hybrid", "rerank")
        rg = ragas_eval(r, eval_set, ragas_modes, n=args.ragas_n)
        print_ragas(rg, min(args.ragas_n, len(eval_set)))


if __name__ == "__main__":
    main()
