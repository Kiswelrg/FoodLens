"""Grab a small, diverse sample of Food-101 photos for eyeballing the VLM.

We DON'T need the full ~5GB Food-101 to validate `describe_image`; the TODO only
asks for ~10 images. This streams the dataset from HuggingFace (no full
download) and saves one photo each from a spread of distinct dishes into
`data/food101_sample/`, named by class so the eyeball test is self-labeling.
"""
from datasets import load_dataset

import config

# A spread that exercises the retrieval blend: exact-token dishes (pad thai),
# semantic/attribute dishes (french onion soup), and visually-similar traps.
WANT = {
    "pad_thai", "french_onion_soup", "caesar_salad", "guacamole",
    "beef_tacos", "chicken_curry", "margherita_pizza", "ramen",
    "chocolate_cake", "grilled_salmon",
}

OUT = config.SAMPLE_DIR


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("ethz/food101", split="train", streaming=True)
    names = ds.features["label"].names
    want_ids = {names.index(w) for w in WANT if w in names}
    missing = WANT - {names[i] for i in want_ids}
    if missing:
        print(f"(not in dataset, skipping: {sorted(missing)})")

    seen: set[int] = set()
    for ex in ds:
        lbl = ex["label"]
        if lbl in want_ids and lbl not in seen:
            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            path = OUT / f"{names[lbl]}.jpg"
            img.save(path, format="JPEG", quality=90)
            seen.add(lbl)
            print(f"saved {path}")
        if len(seen) == len(want_ids):
            break
    print(f"\n{len(seen)} images in {OUT}")


if __name__ == "__main__":
    main()
