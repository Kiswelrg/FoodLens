"""Proof that `recipe_search` is callable FROM INSIDE a LangGraph agent over MCP.

The FoodGraph capstone (image -> answer) reaches RAG through the in-process
bridge for speed. This script shows the *other* half of TODO 4's "done when":
the SAME retriever, now exposed as an MCP tool by server.py, loaded into a
LangGraph ReAct agent via langchain-mcp-adapters and driven by DeepSeek. The
agent decides on its own to call `recipe_search`, then answers grounded on the
returned recipe cards -- exactly how Claude Desktop would use the tool, but in a
LangGraph agent, satisfying the Week-3 "LangGraph + MCP" milestone.

Run:  python agent_demo.py            # default demo query
      python agent_demo.py "high-protein vegetarian dinner under 500 calories"
"""
import asyncio
import sys
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

HERE = Path(__file__).resolve().parent
FOODRAG_DIR = HERE.parent / "FoodRAG"
SERVER = HERE / "server.py"

# Reuse FoodRAG's DeepSeek credential loading (.env, no hardcoded secret). llm.py
# imports nothing named `config`, so adding FoodRAG to the path is collision-safe.
sys.path.insert(0, str(FOODRAG_DIR))
from llm import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL   # noqa: E402


def _deepseek_async(max_tokens: int = 1024) -> ChatOpenAI:
    """DeepSeek chat model with an ASYNC http client (the agent runs via ainvoke)
    and trust_env=False so proxy env vars never hijack the call (matches llm.py)."""
    async_http = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(proxy=None), trust_env=False)
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_async_client=async_http,
        max_tokens=max_tokens,
    )


async def run(query: str) -> str:
    client = MultiServerMCPClient({
        "foodlens": {
            "command": sys.executable,
            "args": [str(SERVER)],
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    print(f"Loaded MCP tools: {[t.name for t in tools]}\n")

    agent = create_react_agent(_deepseek_async(), tools)
    result = await agent.ainvoke({"messages": [("user", query)]})

    # Show the tool-call trace so the MCP round-trip is visible, then the answer.
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"  -> agent called {call['name']}({call['args']})")
    return result["messages"][-1].content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or \
        "Suggest a comforting vegetarian noodle soup and tell me its calories."
    print(f"Query: {q}\n")
    print(asyncio.run(run(q)))
