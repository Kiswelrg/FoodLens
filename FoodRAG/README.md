# FoodRAG — hybrid-search RAG over recipes & nutrition

A standalone, benchmarked RAG pipeline (dense + sparse + RRF fusion +
cross-encoder reranking) over the Food.com recipe corpus. Built as the
retrieval foundation for a later image→VLM→RAG agent, but it stands on its own.

This is **portfolio component #1**: a RAG pipeline you can put numbers on, so the
resume line *"RAG pipelines (hybrid search, reranking, vector databases)"* is
something you can defend in an interview rather than just assert.

## The use case it serves

A "FoodLens" assistant. Eventually: photograph a dish → a vision-language model
describes it → that description becomes the retrieval query → the pipeline
returns the matching recipe, nutrition, and dietary notes → a grounded summary.

The VLM supplies **perception**; this RAG supplies the **facts the VLM can't
reliably know** (real steps, exact calories, substitutions). That division is
why the RAG node is load-bearing, not decorative.

For now we ignore the VLM and build/benchmark the retrieval half with typed
queries. The summary node already takes a `query` string, so wiring the VLM in
later changes nothing here — see [The VLM seam](#the-vlm-seam).

## Architecture

```
            ┌─ dense  (bge embeddings, cosine)   top-50 ─┐
  query ───►┤                                            ├─► RRF fuse ─► rerank ─► top-5 ─► summary
            └─ sparse (BM25)                     top-50 ─┘   (k=60)     (bge-       (LLM,
                                                                        reranker)   grounded)
```

Why each stage earns its place:

- **Dense** handles paraphrase/semantics — *"a comforting cold-weather stew."*
- **Sparse (BM25)** handles exact tokens dense misses — ingredient names,
  `gluten-free`, a specific calorie figure.
- **RRF fusion** merges the two on *rank* (not raw score), sidestepping the
  score-incompatibility problem that breaks naive weighted blending. k=60.
- **Reranking** is a cross-encoder that reads (query, doc) jointly and reorders
  the fused candidates for precision. Retrieval sets the ceiling; reranking
  sharpens what's already in the pool.

→ **`hybrid search` + `reranking` + `vector` retrieval, all visible in the code.**

## Files

| file | role |
|---|---|
| `config.py` | paths, model choices, retrieval knobs |
| `data_loader.py` | Food.com CSV → atomic recipe documents + metadata |
| `index.py` | build & persist dense embeddings + BM25 index + docstore |
| `retrieval.py` | dense / sparse / RRF / rerank / unified `search()` |
| `llm.py` | shared DeepSeek (OpenAI-compatible) client for the eval |
| `eval_set.py` | LLM-generated, leak-checked benchmark queries |
| `ragas_compat.py` | imports RAGAS on a langchain-v1 stack; DeepSeek-backed judge |
| `evaluate.py` | known-item MRR/Hit@k + bootstrap CI + RAGAS, across five modes |
| `summarize.py` | grounded LLM summary (the VLM seam) |

## Setup

```bash
pip install -r requirements.txt

# 1. Download the dataset (Kaggle: "Food.com Recipes and Interactions",
#    user shuyangli94) and place RAW_recipes.csv in ./data/
# 2. Build the indices (downloads the bge models on first run):
python index.py
# 3. Try a query:
python retrieval.py
# 4. Build the benchmark query set (needs a DeepSeek key, see below) and run it:
python eval_set.py        # generates artifacts/eval_set.json (one-time)
python evaluate.py        # known-item + RAGAS; --no-ragas for the fast path
# 5. End-to-end grounded answer (needs Ollama running, or set backend="anthropic"):
python summarize.py
```

The benchmark uses **DeepSeek** (an OpenAI-compatible endpoint) to write the eval
queries and to act as the RAGAS judge. Set `DEEPSEEK_API_KEY` (or edit the
fallback in `llm.py`). `eval_set.json` is cached, so generation runs once;
re-runs of `evaluate.py` reuse it.

First run downloads `bge-base-en-v1.5` (~110M) and, on first rerank,
`bge-reranker-v2-m3` (~560M). On CPU the reranker is slow per doc but only sees
~20 candidates, so it's fine for a dev loop; a GPU makes it instant. For fast
CPU iteration, set `RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`.

## Evaluation

The benchmark is built to be **citable**, not just illustrative. The earlier
version used each recipe's own description as the query — which leaks the answer
into the question and inflates every number. That's been replaced.

**Query set (`eval_set.py`).** DeepSeek writes a natural-language search query
for each of 150 sampled recipes, phrased as a user who *wants* such a dish but
has never seen that recipe. Queries may name a few ingredients or a dietary/time
constraint (real users do — and that's where sparse + hybrid earn their keep),
but a programmatic leak check **drops any query that copies the recipe name or
shares a 5-gram with its description**. 10 leaking queries were dropped,
leaving **132** honest queries. The gold doc is the source recipe (known-item
retrieval).

**Two scorings (`evaluate.py`).**
- *Known-item* — MRR@10 and Hit@{1,5,10}, with a **bootstrap 95% CI** on MRR.
- *RAGAS* — **context precision** (LLM-judged by DeepSeek against a reference
  answer: are genuinely-relevant recipes ranked first?) and **context recall**
  (non-LLM string match: was the gold recipe card retrieved?). RAGAS credits a
  method for surfacing *other* relevant recipes that strict known-item MRR would
  miscount as misses.

`random` is a floor baseline (uniformly sampled docs) to anchor the metric scale.

### Known-item — 132 leak-checked queries

| mode               | MRR@10 | 95% CI         | Hit@1 | Hit@5 | Hit@10 |
|--------------------|:------:|:--------------:|:-----:|:-----:|:------:|
| random (floor)     | 0.008  | [0.000, 0.030] | 0.008 | 0.008 | 0.008  |
| sparse (BM25)      | 0.645  | [0.581, 0.715] | 0.545 | 0.765 | 0.841  |
| dense (bge)        | 0.709  | [0.645, 0.777] | 0.621 | 0.818 | 0.871  |
| hybrid (RRF)       | 0.726  | [0.665, 0.794] | 0.644 | 0.856 | 0.886  |
| **hybrid+rerank**  | **0.782** | [0.727, 0.841] | **0.697** | **0.902** | **0.932** |

### RAGAS — top-5 contexts, DeepSeek judge, 60-query subset

| mode              | context_precision | context_recall |
|-------------------|:-----------------:|:--------------:|
| random (floor)    | 0.000             | 0.000          |
| sparse (BM25)     | 0.632             | 0.717          |
| dense (bge)       | 0.718             | 0.767          |
| hybrid (RRF)      | 0.713             | 0.817          |
| **hybrid+rerank** | **0.819**         | **0.900**      |

**Reading the numbers.** MRR climbs monotonically from each single retriever
(0.645 / 0.709) through fusion (0.726) to reranking (**0.782**) — a **+10%**
relative lift from hybrid over the best single method and **+7.7%** more from the
reranker. The RAGAS split tells *why* each stage helps: **fusion buys recall**
(0.767 → 0.817 — two retrievers catch what one drops) while **reranking buys
precision** (0.713 → 0.819 — the cross-encoder reorders the survivors). Numbers
will shift slightly run-to-run (query generation + the LLM judge are sampled),
but the ordering is stable.

## The VLM seam

`summarize.answer(query, ...)` takes a text `query`. In the Week 3 LangGraph
build, the only change is *where that string comes from*:

```
# now (standalone):     query = "quick high-protein vegetarian dinner"
# week 3 (in LangGraph): query = vlm_node(image_url)   # e.g. "grilled salmon,
#                                                       #       asparagus, lemon"
```

The RAG becomes a single LangGraph node that calls `Retriever.search(...)`; the
VLM node feeds it; the summary node consumes it. Test images: Food-101.

## Productionization (later, optional)

The dense index is currently a numpy array on disk so it runs on a laptop with
zero infra. The clean next step — and a separate resume-able task — is moving
dense vectors into **pgvector** (fits a Django/Postgres stack), keeping BM25 +
RRF + rerank in the app layer. The CSV loads straight into a Postgres table, so
the corpus is already "SQL-shaped."
