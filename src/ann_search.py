import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import faiss
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "two_tower")

TRAINING_DATA = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "training_data.csv"
)


print("Loading models...")

user_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "user_model.keras"),
    compile=False
)

item_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "item_model.keras"),
    compile=False
)

print("Models loaded successfully!")


print("Loading article IDs...")

df = pd.read_csv(
    TRAINING_DATA,
    usecols=["article_id"],
    nrows=1000000
)

df["article_id"] = df["article_id"].astype(str)

article_ids = df["article_id"].unique()

print("Unique articles:", len(article_ids))


print("Generating item embeddings...")

item_embeddings = item_model(
    tf.constant(article_ids)
).numpy()

print("Item embeddings shape:", item_embeddings.shape)


def recommend(customer_id, top_k=10):

    customer_id = str(customer_id)

    user_embedding = user_model(
        tf.constant([customer_id])
    ).numpy().astype("float32")

    scores, indices = faiss_index.search(
        user_embedding,
        top_k
    )

    recommendations = []

    for score, index in zip(scores[0], indices[0]):

        if index != -1:
            recommendations.append({
                "article_id": article_ids[index],
                "score": float(score)
            })

    return recommendations


# -----------------------------------------
# Create FAISS ANN index
# -----------------------------------------

print("\nCreating FAISS ANN index...")

embedding_dimension = item_embeddings.shape[1]

faiss_index = faiss.IndexFlatIP(embedding_dimension)

faiss_index.add(
    item_embeddings.astype("float32")
)

print("FAISS index created successfully!")
print("Indexed items:", faiss_index.ntotal)

# -----------------------------------------
# Save FAISS index
# -----------------------------------------

FAISS_DIR = os.path.join(BASE_DIR, "models", "faiss")

os.makedirs(FAISS_DIR, exist_ok=True)

faiss.write_index(
    faiss_index,
    os.path.join(FAISS_DIR, "article_index.faiss")
)

np.save(
    os.path.join(FAISS_DIR, "article_ids.npy"),
    article_ids
)

print("FAISS index saved successfully!")

print("\nTesting recommendation...")

print(
    recommend(
        customer_id="00000c5e3a",
        top_k=10
    )
)