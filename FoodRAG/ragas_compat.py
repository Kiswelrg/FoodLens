"""RAGAS, made importable on a langchain-v1 stack, plus DeepSeek-backed wrappers.

Why this file exists: the pinned `ragas` release unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI` at module load. That path
was removed in langchain-community 0.4 (the v1 era we run), so `import ragas`
explodes before any metric can be used — even though we never touch Vertex. We
inject a harmless stub for that one symbol *before* importing ragas. (Downgrading
the whole langchain stack instead would break the langchain_openai client the
DeepSeek judge relies on.)

It also wires DeepSeek in as RAGAS's judge LLM and bge as its embeddings, so the
LLM-judged metrics run on the same model family as the rest of the eval.
"""
import sys
import types

# --- the stub: must run before `import ragas` ---
_VERTEX = "langchain_community.chat_models.vertexai"
if _VERTEX not in sys.modules:
    _m = types.ModuleType(_VERTEX)
    _m.ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated
    sys.modules[_VERTEX] = _m

from ragas.dataset_schema import SingleTurnSample          # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper     # noqa: E402
from ragas.llms import LangchainLLMWrapper                  # noqa: E402
from ragas.metrics import (                                 # noqa: E402
    LLMContextPrecisionWithReference,
    NonLLMContextRecall,
)

import config                                               # noqa: E402
from llm import get_deepseek                                # noqa: E402

__all__ = [
    "SingleTurnSample",
    "make_context_precision",
    "make_context_recall",
]


def make_context_precision() -> LLMContextPrecisionWithReference:
    """Context precision: LLM-judged. For each retrieved doc, does it help answer
    the query given the reference answer? Rewards ranking truly-relevant recipes
    high — including relevant ones that aren't the single known-item gold."""
    judge = LangchainLLMWrapper(get_deepseek(max_tokens=1024))
    return LLMContextPrecisionWithReference(llm=judge)


def make_context_recall() -> NonLLMContextRecall:
    """Context recall: non-LLM. Did the retrieved set contain the gold recipe
    card (matched by string similarity to the reference context)? Deterministic,
    no judge needed since we know the gold document."""
    return NonLLMContextRecall()
