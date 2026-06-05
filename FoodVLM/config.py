"""FoodVLM config -- the perception component, independent of FoodRAG.

FoodVLM turns a food photo into a retrieval-ready query string. It is a sibling
of FoodRAG, not a part of it: both are wired together only in the eventual
LangGraph. Keep this self-contained so the VLM can be developed and tested on
its own (the one cross-component touch is the optional --search smoke test in
vlm.py, which reaches into FoodRAG purely to eyeball retrieval quality).
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "food101_sample"   # eyeball images from Food-101
DATA_DIR.mkdir(exist_ok=True)

# Sibling RAG component, used only by the optional integration smoke test.
FOODRAG_DIR = ROOT.parent / "FoodRAG"

# --- Model ---
# InternVL3-8B quantized to 4-bit for Apple Silicon via mlx-vlm; downloaded &
# smoke-tested locally. The VLM only describes the photo -- FoodRAG supplies the
# facts it can't reliably know (real steps, exact calories, substitutions).
VLM_MODEL = "mlx-community/InternVL3-8B-MLX-4bit"
VLM_MAX_TOKENS = 256   # a query, not an essay; keep the description compact
