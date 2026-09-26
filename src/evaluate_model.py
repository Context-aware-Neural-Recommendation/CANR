import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_DATA = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "two_tower"
)


print("Loading trained models...")

user_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "user_model.keras"),
    compile=False
)

item_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "item_model.keras"),
    compile=False
)

print("Models loaded successfully!")


# -----------------------------------------
# Load evaluation sample
# -----------------------------------------

print("\nLoading evaluation data...")

df = pd.read_csv(
    TRAINING_DATA,
    nrows=10000
)

df["customer_id"] = df["customer_id"].astype(str)
df["article_id"] = df["article_id"].astype(str)

print("Evaluation rows:", len(df))


# -----------------------------------------
# Create candidate article list
# -----------------------------------------

candidate_items = df["article_id"].unique()

print("Candidate items:", len(candidate_items))


# -----------------------------------------
# Generate item embeddings
# -----------------------------------------

print("\nGenerating item embeddings...")

item_embeddings = item_model(
    tf.constant(candidate_items)
)

print(
    "Item embedding shape:",
    item_embeddings.shape
)


# -----------------------------------------
# Recall@K
# -----------------------------------------

def recall_at_k(k=10):

    correct = 0
    total = 0

    for _, row in df.iterrows():

        user_id = tf.constant(
            [row["customer_id"]]
        )

        user_embedding = user_model(
            user_id
        )

        scores = tf.matmul(
            user_embedding,
            item_embeddings,
            transpose_b=True
        )

        top_k_indices = tf.math.top_k(
            scores[0],
            k=min(k, len(candidate_items))
        ).indices.numpy()

        recommended_items = {
            candidate_items[i]
            for i in top_k_indices
        }

        if row["article_id"] in recommended_items:
            correct += 1

        total += 1

    return correct / total


print("\nCalculating Recall@10...")

recall = recall_at_k(10)

print(
    f"Recall@10: {recall:.4f}"
)

print("\nModel evaluation completed successfully!")