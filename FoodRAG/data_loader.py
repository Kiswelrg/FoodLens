"""Load and normalize the Food.com recipes dataset into retrieval documents.

Dataset: Kaggle "Food.com Recipes and Interactions" (shuyangli94), file
RAW_recipes.csv. Columns include: name, id, minutes, tags (list-string),
nutrition (list-string), n_steps, steps (list-string), description,
ingredients (list-string), n_ingredients.

CHUNKING DECISION: each recipe is ONE atomic document. Recipes are short,
naturally bounded units; sub-chunking would split ingredients from steps and
hurt retrieval, so the recipe card *is* the chunk. (Chunk when a document is
long and topically mixed; don't chunk when the unit is already small and
self-contained.)
"""
import ast
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config

# nutrition list order in the dataset:
# [calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, sat_fat_PDV, carbs_PDV]
NUTRITION_FIELDS = ["calories", "total_fat", "sugar", "sodium",
                    "protein", "saturated_fat", "carbohydrates"]

DIET_KEYWORDS = {
    "vegetarian": ["vegetarian"],
    "vegan": ["vegan"],
    "gluten_free": ["gluten-free", "gluten free"],
    "low_carb": ["low-carb", "low carb"],
    "dairy_free": ["dairy-free", "dairy free"],
}


def _parse_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


@dataclass
class RecipeDoc:
    id: int
    name: str
    text: str                       # the field we embed / run BM25 over
    metadata: dict = field(default_factory=dict)


def _diet_flags(tags: list[str]) -> dict:
    joined = " ".join(str(t) for t in tags).lower()
    return {flag: any(k in joined for k in kws) for flag, kws in DIET_KEYWORDS.items()}


def _to_document(row: dict) -> "RecipeDoc | None":
    name = (str(row.get("name") or "")).strip()
    if not name:
        return None

    tags = _parse_list(row.get("tags"))
    ingredients = _parse_list(row.get("ingredients"))
    steps = _parse_list(row.get("steps"))
    nutrition = _parse_list(row.get("nutrition"))
    desc = (str(row.get("description") or "")).strip()

    nut = {f: (nutrition[i] if i < len(nutrition) else None)
           for i, f in enumerate(NUTRITION_FIELDS)}

    # Human-readable card. Good for BM25 (exact tokens: ingredient names, diet
    # tags, calorie numbers) AND dense (the semantic description).
    parts = [f"Recipe: {name}"]
    if desc:
        parts.append(f"Description: {desc}")
    if ingredients:
        parts.append("Ingredients: " + ", ".join(map(str, ingredients)))
    if tags:
        parts.append("Tags: " + ", ".join(map(str, tags)))
    if nut.get("calories") is not None:
        parts.append(f"Calories: {nut['calories']}")
    if steps:
        parts.append("Steps: " + " ".join(map(str, steps)))
    text = "\n".join(parts)

    meta = {
        "name": name,
        "description": desc,
        "minutes": row.get("minutes"),
        "n_ingredients": row.get("n_ingredients"),
        **{f"nutrition_{k}": v for k, v in nut.items()},
        **_diet_flags(tags),
        "tags": tags,
    }
    return RecipeDoc(id=int(row["id"]), name=name, text=text, metadata=meta)


def load_documents(limit: "int | None" = None) -> list[RecipeDoc]:
    limit = limit or config.CORPUS_LIMIT
    df = pd.read_csv(config.RECIPES_CSV, nrows=limit)
    docs: list[RecipeDoc] = []
    for _, row in df.iterrows():
        d = _to_document(row.to_dict())
        if d is not None:
            docs.append(d)
    return docs


if __name__ == "__main__":
    docs = load_documents(limit=200)
    print(f"Loaded {len(docs)} recipe documents")
    print("---- sample document text ----")
    print(docs[0].text[:500])
    print("---- sample metadata ----")
    print({k: docs[0].metadata[k] for k in list(docs[0].metadata)[:8]})
