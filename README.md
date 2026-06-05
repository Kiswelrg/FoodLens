# FoodLens

Photograph a dish, get a grounded answer about it. A vision-language model
describes the photo; that description becomes a search query into a hybrid
retrieval pipeline over ~180k real recipes; an LLM writes the final answer using
**only** the retrieved recipes — real steps, exact calories, dietary notes.

The division of labor is the whole point:

> **the VLM supplies perception; the RAG supplies the facts the VLM can't
> reliably know.** A vision model can tell *pad thai* from *ramen*, but it can't
> recall a recipe's true ingredient list or per-serving calories. Retrieval does.

```
 photo ─► VLM ─► "pad thai; ingredients: rice noodles, …; approx 200 g"
                              │  (this string is the retrieval query)
                              ▼
              dense (bge) ─┐
                            ├─► RRF fuse ─► cross-encoder rerank ─► top recipes
              sparse (BM25)─┘
                              │
                              ▼
                   grounded LLM answer  ("asian pad thai [1] — 471 cal/serving …")
```

## Architecture

Three decoupled components, each independently runnable, wired together by a
LangGraph state machine:

| component | role | key tech |
|---|---|---|
| **[FoodVLM/](FoodVLM/)** | perception — photo → compact retrieval query | InternVL3-8B (4-bit) via MLX |
| **[FoodRAG/](FoodRAG/)** | facts — hybrid retrieval, fusion, reranking, eval | bge embeddings, BM25, RRF, bge-reranker, DeepSeek |
| **[FoodGraph/](FoodGraph/)** | orchestration — the end-to-end graph | LangGraph (+ LangSmith tracing) |

```
StateGraph: { image_path, vlm_description, retrieved_docs, top_score, answer }

  vlm_node ──► rag_node ──► summarize_node ──► END
     │            │  top_score ≥ 0.3
     │            └────────► clarify_node ──► END     (low confidence → ask the user)
```

The VLM and RAG components keep their own top-level `config` modules and never
cross-import; `FoodGraph/bridge.py` co-loads both in one process (isolating the
clashing `config` name) so the whole run traces as a single LangSmith trace.

## The retrieval pipeline

`dense (top-50) + sparse (top-50) → RRF fuse (top-20) → cross-encoder rerank
(top-5)`. Each stage is a separate, independently benchmarkable method:

- **Dense** (`bge-base-en-v1.5`, cosine) catches semantics — *"a comforting
  cold-weather stew."*
- **Sparse** (BM25) catches exact tokens — ingredient names, `gluten-free`, a
  calorie number.
- **RRF** fuses the two on *ranks* (not raw scores, which are incomparable),
  hand-written so the fusion stays explicit.
- **Reranker** (`bge-reranker-v2-m3`) reorders the survivors with a
  cross-encoder, lazy-loaded since it's heavy.

Recipes are stored **one document per recipe** (atomic) — short, bounded units
where sub-chunking would split ingredients from steps and hurt retrieval.

## Benchmark

A leak-checked, known-item benchmark over **132** DeepSeek-generated natural
queries (queries that copy the recipe name or share a 5-gram with its
description are dropped, so the answer never leaks into the question). Full
methodology and the RAGAS table are in [FoodRAG/README.md](FoodRAG/README.md).

| mode | MRR@10 | 95% CI | Hit@1 | Hit@5 | Hit@10 |
|---|:--:|:--:|:--:|:--:|:--:|
| sparse (BM25) | 0.645 | [0.581, 0.715] | 0.545 | 0.765 | 0.841 |
| dense (bge) | 0.709 | [0.645, 0.777] | 0.621 | 0.818 | 0.871 |
| hybrid (RRF) | 0.726 | [0.665, 0.794] | 0.644 | 0.856 | 0.886 |
| **hybrid+rerank** | **0.782** | [0.727, 0.841] | **0.697** | **0.902** | **0.932** |

**Fusion buys recall, reranking buys precision:** RAGAS context recall climbs
0.767 → 0.817 when the second retriever is added (it catches what one drops),
and context precision climbs 0.713 → 0.819 when the cross-encoder reorders the
survivors.

## Low-confidence handling

When the VLM misreads a dish, retrieval scores **collapse** rather than returning
confident garbage — a usable signal. FoodGraph turns it into a graph branch: if
the top rerank score falls below a threshold, the graph asks the user to clarify
instead of summarizing a likely-wrong recipe.

| photo | VLM read | top hit | outcome |
|---|---|---|---|
| pad thai | Pad Thai | 0.994 *asian pad thai* | grounded answer |
| guacamole | Guacamole | 0.967 *basic guacamole* | grounded answer |
| french onion soup | "Bao buns" (misread) | 0.188 | → asks the user |

## Quickstart

Each component has its own `requirements.txt`. The dataset and the built indices
are **not** checked in (they're large/generated — see below).

```bash
# 1. RAG: drop Food.com RAW_recipes.csv into FoodRAG/data/, then build the index
cd FoodRAG && pip install -r requirements.txt && python index.py

# 2. VLM: grab a few sample photos to try (Apple Silicon / MLX)
cd ../FoodVLM && pip install -r requirements.txt && python fetch_samples.py

# 3. End to end: photo in, grounded answer out
cd ../FoodGraph && pip install -r requirements.txt
python run.py ../FoodVLM/data/food101_sample/pad_thai.jpg -v
```

### Configuration

- **LLM** (query generation, RAGAS judge, the grounded answer) is DeepSeek via an
  OpenAI-compatible endpoint. Set `DEEPSEEK_API_KEY` in your environment.
- **Tracing** is opt-in: set `LANGSMITH_API_KEY` to record runs to LangSmith
  (project `FoodLens`); without it the graph runs untraced.

```bash
export DEEPSEEK_API_KEY=sk-...
export LANGSMITH_API_KEY=ls__...   # optional
```

## Repository layout

```
FoodLens/
├── FoodVLM/     perception: describe_image(path) -> retrieval query
├── FoodRAG/     retrieval + fusion + rerank + leak-checked benchmark
├── FoodGraph/   LangGraph: image path in -> grounded answer out
└── README.md
```

## Data sources

- **Recipes:** Food.com `RAW_recipes.csv` (~180k recipes) — the retrieval corpus.
- **Test photos:** Food-101 — streamed in small samples, no full download needed.

## Not in version control

The dataset, built indices, and model weights are large and reproducible, so
they're git-ignored (rebuild with `python index.py`). Provide credentials via
environment variables, not source — see Configuration above.
