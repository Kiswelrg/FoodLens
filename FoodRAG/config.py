from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"            # put RAW_recipes.csv here
INDEX_DIR = ROOT / "artifacts"      # built indices land here
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

RECIPES_CSV = DATA_DIR / "RAW_recipes.csv"   # Food.com dataset

# --- Models ---
# Dense embedder. bge-base-en-v1.5 is fast and CPU-friendly for English recipes.
# Swap to "BAAI/bge-m3" if you want multilingual (heavier, ~560M).
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# bge-v1.5 wants this instruction prepended to QUERIES only (never documents).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Cross-encoder reranker. The resume-target model (multilingual, Apache-2.0,
# ~560M, runs on a consumer GPU; ~300ms/doc on CPU so fine for ~20 candidates).
# For fast CPU iteration swap to "cross-encoder/ms-marco-MiniLM-L-6-v2".
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Retrieval knobs ---
CORPUS_LIMIT = 20000   # cap recipes while developing; raise later
DENSE_TOPK = 50        # candidates from dense retriever
SPARSE_TOPK = 50       # candidates from sparse retriever
RRF_K = 60             # reciprocal rank fusion constant
FUSED_TOPK = 20        # candidates passed into the reranker
FINAL_TOPK = 5         # results returned after reranking
