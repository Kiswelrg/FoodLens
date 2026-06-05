"""Generation node: turn retrieved recipes into a grounded answer.

THIS IS THE SEAM WHERE THE VLM PLUGS IN LATER. For the standalone RAG, `query`
is typed text. In the Week 3 LangGraph project, the VLM node will look at a food
image, produce a description string, and THAT becomes `query` here -- nothing
else in this file changes.
"""
import config
from retrieval import Retriever

SYSTEM = (
    "You are a cooking assistant. Answer using ONLY the retrieved recipes below. "
    "Cite recipes by name. Give nutrition figures only if present in the context. "
    "If the context doesn't cover the question, say so rather than inventing details."
)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h['name']}\n{h['text'][:800]}")
    return "\n\n".join(blocks)


def summarize_anthropic(query: str, hits: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Question: {query}\n\nRetrieved recipes:\n{build_context(hits)}"}],
    )
    return msg.content[0].text


def summarize_ollama(query: str, hits: list[dict], model: str = "qwen2.5:7b") -> str:
    # No API key: talk to local Ollama's OpenAI-compatible endpoint.
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Question: {query}\n\nRetrieved recipes:\n{build_context(hits)}"},
        ],
    )
    return resp.choices[0].message.content


def summarize_deepseek(query: str, hits: list[dict]) -> str:
    # The project's default summary backend (no local Ollama / Anthropic key
    # needed): DeepSeek via the shared OpenAI-compatible client in llm.py.
    from langchain_core.messages import HumanMessage, SystemMessage

    from llm import get_deepseek
    llm = get_deepseek(temperature=0.0, max_tokens=600)
    resp = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Question: {query}\n\n"
                             f"Retrieved recipes:\n{build_context(hits)}"),
    ])
    return resp.content


_BACKENDS = {
    "deepseek": summarize_deepseek,
    "anthropic": summarize_anthropic,
    "ollama": summarize_ollama,
}


def summarize(query: str, hits: list[dict], backend: str = "deepseek") -> str:
    """Grounded answer over ALREADY-retrieved hits (no re-retrieval).

    This is the function the LangGraph `summarize_node` calls: the graph's
    rag_node has already produced `hits`, so we don't want `answer()` to fetch
    them a second time.
    """
    return _BACKENDS[backend](query, hits)


def answer(query: str, backend: str = "deepseek") -> str:
    r = Retriever()
    hits = r.search(query, mode="rerank")
    return summarize(query, hits, backend=backend)


if __name__ == "__main__":
    print(answer("what's a quick high-protein vegetarian dinner?", backend="ollama"))
