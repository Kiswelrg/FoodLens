# FoodGraph — the LangGraph capstone

Wires the two sibling components into one traced pipeline:

```
        image_path
            │
        ┌───▼────┐   FoodVLM.describe_image      (perception)
        │vlm_node│   photo → "<dish>; ingredients: …; attributes: …; approx N g"
        └───┬────┘
            │ vlm_description (becomes the retrieval query)
        ┌───▼────┐   FoodRAG.Retriever.search    (facts: dense+sparse→RRF→rerank)
        │rag_node│   → retrieved_docs, top_score
        └───┬────┘
   top_score │ ≥ 0.3 ?  ── no ──►  ┌────────────┐
            │ yes                  │clarify_node│  "not confident, send a clearer photo"
        ┌───▼─────────┐            └─────┬──────┘
        │summarize_node│  FoodRAG.summarize (grounded DeepSeek answer)
        └───┬─────────┘                  │
            └────────────► END ◄──────────┘
```

`StateGraph` state: `{ image_path, vlm_description, retrieved_docs, top_score, answer }`.

FoodGraph owns **no models or data** — it imports the seams and orchestrates
them. The division of labor it realizes: **the VLM supplies perception; the RAG
supplies the facts the VLM can't reliably know.**

## Why a `bridge.py`

FoodVLM and FoodRAG are deliberately decoupled siblings, and *each has its own
top-level `config` module*. The standalone `FoodVLM/vlm.py --search` smoke test
kept them apart by running FoodRAG in a **subprocess**. The capstone instead
needs both **in one process** so LangSmith traces the whole run as one trace.

`bridge.py` is the in-process counterpart of that subprocess isolation: only the
name `config` actually collides, so it imports each component with that
component's dir at the front of `sys.path` and the cached `config` evicted, then
hands FoodGraph's own `config` back. The components stay untouched. See
[`bridge.py`](bridge.py).

## The conditional edge

The eyeball test found an honest failure mode: when the VLM misreads a dish, the
top rerank score **collapses** instead of returning confident garbage. FoodGraph
turns that into a graph branch — if `top_score < CONFIDENCE_THRESHOLD` (0.3), it
routes to `clarify_node` ("ask the user") rather than summarizing a likely-wrong
recipe. Verified on real photos:

| photo | VLM read | top hit | branch |
|---|---|---|---|
| `pad_thai.jpg` | Pad Thai | 0.994 *asian pad thai* | → grounded answer |
| `guacamole.jpg` | Guacamole | 0.967 *basic guacamole* | → grounded answer |
| `french_onion_soup.jpg` | "Bao buns" (misread) | 0.188 | → **clarify** |

## Usage

```bash
pip install -r requirements.txt          # plus FoodVLM/ and FoodRAG/ requirements
python run.py ../FoodVLM/data/food101_sample/pad_thai.jpg -v
python run.py ../FoodVLM/data/food101_sample/*.jpg        # batch (10+ traceable runs)
```

```python
from graph import build_graph
graph = build_graph()                     # builds the Retriever once, reuses it
state = graph.invoke({"image_path": "photo.jpg"})
print(state["answer"])
```

## LangSmith tracing

Tracing is **env-gated** so the graph runs with or without an account. To record
runs (project `FoodLens`):

```bash
export LANGSMITH_API_KEY=ls__...          # your LangSmith key
python run.py ../FoodVLM/data/food101_sample/*.jpg
```

`config.enable_langsmith()` flips on `LANGCHAIN_TRACING_V2` when a key is present;
LangGraph then traces each node automatically. Without a key, `run.py` prints
`LangSmith tracing: off` and proceeds untraced.

## Files

| file | role |
|---|---|
| `config.py` | orchestration knobs (retrieval mode, summary backend, confidence threshold, LangSmith toggle) |
| `bridge.py` | loads FoodVLM + FoodRAG seams in one process despite the shared `config` name |
| `graph.py` | the `StateGraph`: `vlm_node → rag_node → {summarize_node \| clarify_node} → END` |
| `run.py` | CLI: image path(s) → grounded answer(s); enables tracing if a key is set |

## Notes

- `MemorySaver` is not wired yet — the pipeline is single-shot (image → answer).
  Add it when FoodGraph becomes conversational (state already a `TypedDict`).
- Next up (TODO 4): expose `Retriever.search` as an MCP tool so the same
  retrieval is callable from Claude Desktop and from inside this graph.
