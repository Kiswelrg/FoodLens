# FoodMCP — the FoodLens RAG, exposed as an MCP tool

The fourth sibling component (next to `FoodRAG/`, `FoodVLM/`, `FoodGraph/`). It
wraps FoodRAG's three-stage retriever as a **single MCP tool**, `recipe_search`,
served over **stdio** with FastMCP. Owns no models or data of its own — it
imports `FoodRAG.Retriever` and re-publishes `.search` over the Model Context
Protocol so any MCP client can query the corpus.

## Why RAG and not the VLM

MCP's value is giving a client a capability it *can't already do*. An MCP client
that has its own vision model gains nothing from wrapping our local InternVL3-8B.
What a client **cannot** do is reach the private 180k-recipe corpus with hybrid
search + RRF + cross-encoder reranking. That retrieval is the load-bearing
fact-supplier in FoodLens, so it is the seam worth exposing. (RAG-only was the
chosen scope for TODO 4.)

## The tool

`recipe_search(query: str, mode: str = "rerank", k: int = 5) -> list[dict]`

- **mode**: `rerank` (full dense+BM25 → RRF → rerank, default), `hybrid` (fusion,
  no rerank), `dense` (semantic), `sparse` (BM25).
- **returns**: best-first records, each `{name, score, minutes, calories,
  n_ingredients, diets, card}`. `card` is the human-readable recipe text to
  ground an answer on.

## Files

| file | role |
|---|---|
| `server.py` | FastMCP stdio server; defines and serves `recipe_search`. Knobs are inline constants (no `config.py`, to avoid the FoodRAG `import config` collision). |
| `agent_demo.py` | a **LangGraph ReAct agent** (DeepSeek) that loads `recipe_search` over MCP via `langchain-mcp-adapters` and answers grounded — the working proof that the tool is callable from inside a LangGraph agent. |
| `requirements.txt` | adds `mcp[cli]` (server) on top of FoodRAG's deps; `agent_demo.py` also needs `langchain-mcp-adapters`. |

## Run

Install (FoodRAG deps must already be installed, and its `artifacts/` built):

```bash
pip install -r FoodMCP/requirements.txt
pip install langchain-mcp-adapters      # only for agent_demo.py
```

**Standalone server (stdio):**
```bash
python FoodMCP/server.py
```

**Inspect interactively** (MCP Inspector — lists the tool, lets you call it):
```bash
mcp dev FoodMCP/server.py
```

**LangGraph-agent-over-MCP demo** (the in-agent callability proof):
```bash
python FoodMCP/agent_demo.py "high-protein vegetarian dinner under 500 calories"
```
The agent autonomously calls `recipe_search` (often refining the query across a
few calls) and returns an answer grounded on real corpus calories.

## Wiring into any stdio MCP client

Any MCP client that launches stdio servers (the MCP Inspector, a custom agent,
desktop LLM apps) takes the same launch spec — an absolute interpreter path and
an absolute path to `server.py` (the server has no project cwd to rely on):

```json
{
  "command": "/Users/wxy/miniconda3/envs/ams/bin/python",
  "args": ["/Users/wxy/Projects/Job/June2026/Code/FoodLens/FoodMCP/server.py"]
}
```

`agent_demo.py` builds exactly this spec programmatically via
`MultiServerMCPClient` — the canonical way to hand the tool to a LangGraph agent.

## Notes

- **First call loads heavy models** (bge embedder + bge-reranker); the server
  keeps them resident, so subsequent calls are fast. Loading is lazy, so the
  server starts instantly and only pays the cost on first `recipe_search`.
- **No config collision**: this server imports only FoodRAG. `server.py` pins
  FoodRAG's dir at the front of `sys.path` and evicts any cached `config` before
  importing `retrieval`, so FoodRAG's bare `import config` resolves correctly —
  the same discipline as `FoodGraph/bridge.py`, but with a single component so
  there's no clash to referee.
- The **FoodGraph capstone** still reaches RAG through the in-process bridge (no
  subprocess) for latency; FoodMCP is the *externally callable* face of the same
  retriever — for standalone MCP clients and agents.
