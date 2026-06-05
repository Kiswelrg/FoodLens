"""Load FoodVLM and FoodRAG in one process despite their clashing `config`.

FoodVLM and FoodRAG are decoupled sibling components, and each has its OWN
top-level `config` module (FoodRAG additionally has `retrieval`, `summarize`,
`llm`; FoodVLM has `vlm`). The standalone `FoodVLM/vlm.py --search` smoke test
sidestepped the clash by running FoodRAG in a SUBPROCESS. The LangGraph capstone
wants both in ONE process so LangSmith can trace the whole pipeline as a single
run.

Only ONE name actually collides: `config`. `retrieval`, `summarize`, `llm`, and
`vlm` are unique, so once imported they can stay cached -- which matters because
some seams import lazily at call time (e.g. `summarize_deepseek` does
`from llm import get_deepseek`), so their dirs must remain importable at runtime.

So the strategy is:
  * keep both component dirs permanently on `sys.path` (runtime lazy imports),
  * import each component's modules in a phase where THAT component's dir is at
    the front of `sys.path` and the cached `config` has been evicted, so the
    bare `import config` inside each module resolves to the right one,
  * restore the FoodGraph `config` in `sys.modules` at the end so the rest of
    FoodGraph keeps seeing its own config.
The already-imported modules keep their own `config` object bound in their
globals, so runtime calls stay correct regardless of `sys.modules['config']`.

This keeps the components untouched (no edits to their bare imports) while making
them co-resident -- the in-process counterpart of the subprocess isolation.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOODVLM_DIR = ROOT / "FoodVLM"
FOODRAG_DIR = ROOT / "FoodRAG"

# Permanent: lets runtime lazy imports (e.g. FoodRAG's `from llm import ...`)
# resolve. Appended (not prepended) so FoodGraph's own modules keep priority.
for _d in (str(FOODRAG_DIR), str(FOODVLM_DIR)):
    if _d not in sys.path:
        sys.path.append(_d)


def _load_component(component_dir: Path, module_names: list[str]) -> dict:
    """Import `module_names` from `component_dir` with that dir at the front of
    `sys.path` and the shared `config` evicted, so each module's bare
    `import config` binds THIS component's config. Returns {name: module}."""
    sys.modules.pop("config", None)          # force a fresh, dir-correct config
    sys.path.insert(0, str(component_dir))   # priority for this phase's imports
    try:
        return {name: importlib.import_module(name) for name in module_names}
    finally:
        sys.path.remove(str(component_dir))  # drop the priority copy (perm copy stays)


# Preserve FoodGraph's own `config` so we can hand it back after the phases.
_foodgraph_config = sys.modules.get("config")

_vlm = _load_component(FOODVLM_DIR, ["vlm"])
_rag = _load_component(FOODRAG_DIR, ["retrieval", "summarize", "llm"])

# Restore FoodGraph's config as the live `config` module for the rest of the app.
if _foodgraph_config is not None:
    sys.modules["config"] = _foodgraph_config
else:
    sys.modules.pop("config", None)

# --- the seam callables the graph wraps ---------------------------------------
describe_image = _vlm["vlm"].describe_image
Retriever = _rag["retrieval"].Retriever
summarize = _rag["summarize"].summarize

__all__ = ["describe_image", "Retriever", "summarize"]
