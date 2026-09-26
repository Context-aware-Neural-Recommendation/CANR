import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = "models/two_tower"
DATA_PATH = "data/features/training_data.csv"

EVAL_ROWS = 10000
K = 10

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

print("\nLoading evaluation data...")

df = pd.read_csv(DATA_PATH, nrows=EVAL_ROWS)

print(f"Evaluation rows: {len(df)}")

candidate_items = df["article_id"].astype(str).to_numpy(dtype="object")

print(f"Candidate items: {len(candidate_items)}")

print("\nGenerating item embeddings...")

item_embeddings = item_model.predict(
    candidate_items,
    batch_size=256,
    verbose=0
)

print(f"Item embedding shape: {item_embeddings.shape}")

print("\nCalculating NDCG@10...")

ndcg_scores = []

for _, row in df.iterrows():

    user_id = str(row["customer_id"])
    actual_item = str(row["article_id"])

    user_embedding = user_model.predict(
        np.array([user_id]),
        verbose=0
    )[0]

    scores = np.dot(item_embeddings, user_embedding)

    top_k_indices = np.argsort(scores)[-K:][::-1]

    recommended_items = candidate_items[top_k_indices]

    if actual_item in recommended_items:

        rank = np.where(recommended_items == actual_item)[0][0] + 1

        ndcg = 1 / np.log2(rank + 1)

    else:

        ndcg = 0

    ndcg_scores.append(ndcg)

ndcg_at_k = np.mean(ndcg_scores)

print(f"NDCG@10: {ndcg_at_k:.4f}")

print("\nNDCG evaluation completed successfully!")