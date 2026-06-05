"""Build and persist the dense (embedding) and sparse (BM25) indices.

Run once after downloading RAW_recipes.csv:  python index.py
"""
import json

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from data_loader import RecipeDoc, load_documents


def _embed_documents(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Documents are embedded WITHOUT the query instruction (bge convention).
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def build(limit: "int | None" = None) -> None:
    docs: list[RecipeDoc] = load_documents(limit)
    texts = [d.text for d in docs]
    ids = [d.id for d in docs]
    print(f"Building indices for {len(docs)} documents")

    # --- Dense ---
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = _embed_documents(model, texts)
    np.save(config.INDEX_DIR / "dense_embeddings.npy", embeddings)

    # --- Sparse (BM25) ---
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(config.INDEX_DIR / "bm25_index"))

    # --- Doc store (aligned with the same row order as the indices) ---
    store = [{"id": d.id, "name": d.name, "text": d.text, "metadata": d.metadata}
             for d in docs]
    with open(config.INDEX_DIR / "docstore.json", "w") as f:
        json.dump({"ids": ids, "docs": store}, f)

    print(f"Saved dense_embeddings.npy, bm25_index/, docstore.json -> {config.INDEX_DIR}")


if __name__ == "__main__":
    build()
