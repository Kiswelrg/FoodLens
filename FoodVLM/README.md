# FoodVLM — perception component

Turns a food photo into a **compact, retrieval-ready query string**. It is a
sibling of [`FoodRAG`](../FoodRAG), not a part of it; the two are wired together
only in the eventual LangGraph build. The division of labor:

> **the VLM supplies perception; the RAG supplies the facts the VLM can't
> reliably know** (real steps, exact calories, substitutions).

## What it does

```
food photo ──► InternVL3-8B (4-bit, MLX) ──► one structured line:
                                             "<dish>; ingredients: <…>; attributes: <…>; approx <N> g"
```

That line is engineered to be a *good RAG query*, not a caption: dense with the
exact tokens BM25 keys on (dish + ingredient names) plus the salient attributes
the dense embedder rewards. A flowery caption would dilute both retrievers.

## Files

| file | role |
|---|---|
| `config.py` | model choice (`InternVL3-8B-MLX-4bit`), paths, sibling-RAG pointer |
| `vlm.py` | `describe_image(path) -> str`; the future LangGraph `vlm_node` |
| `fetch_samples.py` | stream ~10 diverse Food-101 photos for the eyeball test |

## Usage

```python
from vlm import describe_image
query = describe_image("data/food101_sample/pad_thai.jpg")
# -> "Pad Thai; ingredients: rice noodles, bean sprouts, tofu, ...; attributes: stir-fried; approx 200 g"
```

CLI, including the optional cross-component retrieval eyeball (runs FoodRAG in a
subprocess so the two identically-named `config` modules never collide):

```bash
python fetch_samples.py                       # grab Food-101 eyeball images
python vlm.py data/food101_sample/*.jpg --search
```

## Eyeball results (8 Food-101 photos)

`describe_image` reliably yields RAG-usable strings; fed into
`FoodRAG.Retriever.search` (rerank mode):

- **Strong:** pad thai → *asian pad thai* (0.994), guacamole → *basic guacamole*
  (0.967), chocolate cake → *beatty's chocolate cake* (0.982), grilled salmon,
  chicken noodle soup — all retrieve the right dish family.
- **Honest failure mode:** when the VLM misidentifies a dish (french onion soup
  read as "bao buns"), retrieval scores collapse (top hit 0.188) rather than
  confidently returning a wrong recipe — a useful low-confidence signal for the
  planned LangGraph conditional edge ("if VLM confidence is low, ask the user").

## Notes

- Inference is deterministic (`temperature=0`) so a dish yields a stable query,
  keeping downstream retrieval reproducible.
- The model is lazy-loaded once and cached at module scope (heavy to load).
- Apple-Silicon only (MLX). Swap `VLM_MODEL` in `config.py` for Qwen2.5-VL etc.
