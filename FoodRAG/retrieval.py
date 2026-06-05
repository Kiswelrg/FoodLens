"""Dense + sparse retrieval, RRF fusion, and cross-encoder reranking.

The full three-stage pipeline:
    dense (top-50) + sparse (top-50)  ->  RRF fuse (top-20)  ->  rerank (top-5)

Each stage is a separate method so you can benchmark them in isolation
(evaluate.py) and explain each one independently in an interview.
"""
import json

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config


class Retriever:
    def __init__(self) -> None:
        self.embeddings = np.load(config.INDEX_DIR / "dense_embeddings.npy")
        self.bm25 = bm25s.BM25.load(str(config.INDEX_DIR / "bm25_index"),
                                    load_corpus=False)
        with open(config.INDEX_DIR / "docstore.json") as f:
            store = json.load(f)
        self.ids: list[int] = store["ids"]
        self.docs: list[dict] = store["docs"]          # aligned with index rows
        self.embed_model = SentenceTransformer(config.EMBED_MODEL)
        self._reranker: "CrossEncoder | None" = None   # lazy: heavy to load

    # ---------- single-method retrievers: return [(row_idx, score), ...] ----------
    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embed_model.encode([config.QUERY_INSTRUCTION + query],
                                    normalize_embeddings=True)[0]
        sims = self.embeddings @ q                      # cosine (vectors normalized)
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        q_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self.bm25.retrieve(q_tokens, k=min(k, len(self.docs)),
                                             show_progress=False)
        return [(int(results[0][r]), float(scores[0][r]))
                for r in range(results.shape[1])]

    # ---------- fusion ----------
    @staticmethod
    def rrf(rank_lists: list[list[tuple[int, float]]],
            k: int = config.RRF_K) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion: operates on RANKS, not raw scores, so it
        sidesteps the score-incompatibility problem that breaks naive weighting."""
        fused: dict[int, float] = {}
        for ranked in rank_lists:
            for rank, (idx, _score) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    # ---------- reranker (cross-encoder) ----------
    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def rerank(self, query: str, idxs: list[int], top_n: int) -> list[tuple[int, float]]:
        pairs = [(query, self.docs[i]["text"]) for i in idxs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(idxs, scores), key=lambda x: -float(x[1]))
        return [(int(i), float(s)) for i, s in ranked[:top_n]]

    # ---------- full pipeline ----------
    def search(self, query: str, mode: str = "rerank",
               final_k: "int | None" = None) -> list[dict]:
        """mode in {dense, sparse, hybrid, rerank}."""
        final_k = final_k or config.FINAL_TOPK
        if mode == "dense":
            hits = self.dense(query, final_k)
        elif mode == "sparse":
            hits = self.sparse(query, final_k)
        elif mode in ("hybrid", "rerank"):
            d = self.dense(query, config.DENSE_TOPK)
            s = self.sparse(query, config.SPARSE_TOPK)
            fused = self.rrf([d, s])[:config.FUSED_TOPK]
            if mode == "hybrid":
                hits = fused[:final_k]
            else:
                hits = self.rerank(query, [i for i, _ in fused], final_k)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        return [{**self.docs[i], "score": score} for i, score in hits]


if __name__ == "__main__":
    r = Retriever()
    q = "a comforting spicy vegetarian noodle soup"
    print(f"Query: {q}\n")
    for hit in r.search(q, mode="rerank"):
        print(f"{hit['score']:.3f}  {hit['name']}")
