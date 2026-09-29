import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import numpy as np
import tf_keras


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "exported",
    "item_model.keras"
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "vocabularies.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "embeddings"
)

EMBEDDINGS_PATH = os.path.join(
    OUTPUT_DIR,
    "item_embeddings.npy"
)

ARTICLE_IDS_PATH = os.path.join(
    OUTPUT_DIR,
    "article_ids.npy"
)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading item model...")

    item_model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Item model loaded successfully.")

    print("Loading article vocabulary...")

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        vocabularies = json.load(file)

    article_ids = np.array(
        vocabularies["article_id"],
        dtype=str
    )

    print("Total articles:", len(article_ids))

    print("\nGenerating item embeddings...")

    item_embeddings = item_model(
    article_ids
    ).numpy().astype("float32")

    print(
        "Item embeddings shape:",
        item_embeddings.shape
    )

    np.save(
        EMBEDDINGS_PATH,
        item_embeddings
    )

    np.save(
        ARTICLE_IDS_PATH,
        article_ids
    )

    print("\nItem embeddings exported successfully!")

    print("Embeddings:", EMBEDDINGS_PATH)
    print("Article IDs:", ARTICLE_IDS_PATH)


if __name__ == "__main__":
    main()