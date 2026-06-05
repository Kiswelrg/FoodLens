"""Shared DeepSeek (OpenAI-compatible) client used by the eval pipeline.

DeepSeek is used instead of the Anthropic API for the offline benchmark work
(query generation + RAGAS judging). It speaks the OpenAI protocol, so we drive
it through `langchain_openai.ChatOpenAI`. NOTE: when we need strictly-structured
output we say the word "JSON" and spell out the shape in the prompt — DeepSeek's
JSON mode requires the literal token "json" to appear in the messages.
"""
import os
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency).

    Loads KEY=VALUE lines from `path` into os.environ. A real environment
    variable always wins (setdefault), and the load is keyed off llm.py's own
    location, so it works no matter the cwd — FoodRAG run rooted at its own dir
    (vlm.py's `--search` subprocess) or imported in-process by FoodGraph/bridge.py.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent / ".env")

# No fallback secret in source: the key lives only in FoodRAG/.env (gitignored)
# or the real environment. See FoodRAG/.env.example.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. Copy FoodRAG/.env.example to FoodRAG/.env "
        "and fill in your key, or export DEEPSEEK_API_KEY in your environment."
    )
DEEPSEEK_BASE_URL = "https://api.deepseek.com/"

# trust_env=False so corporate/local proxy env vars never hijack the call.
_http_client = httpx.Client(
    transport=httpx.HTTPTransport(proxy=None),
    trust_env=False,
)


def get_deepseek(temperature: float = 0.0, max_tokens: int = 4096,
                 json_mode: bool = False) -> ChatOpenAI:
    model_kwargs: dict = {"extra_body": {"think": False}}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=_http_client,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )
