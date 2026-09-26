import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import faiss
import tensorflow as tf
from fastapi import FastAPI


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "two_tower"
)

FAISS_DIR = os.path.join(
    BASE_DIR,
    "models",
    "faiss"
)


app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description="Two-Tower Neural Recommendation API with FAISS",
    version="1.0.0"
)


print("Loading user model...")

user_model = tf.keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "user_model.keras"
    ),
    compile=False
)

print("User model loaded successfully!")


print("Loading FAISS index...")

faiss_index = faiss.read_index(
    os.path.join(
        FAISS_DIR,
        "article_index.faiss"
    )
)

article_ids = np.load(
    os.path.join(
        FAISS_DIR,
        "article_ids.npy"
    ),
    allow_pickle=True
)

print("FAISS index loaded successfully!")
print("Indexed items:", faiss_index.ntotal)


@app.get("/")
def home():

    return {
        "message": "Context-Aware Neural Recommendation Engine API is running"
    }


@app.get("/recommend/{customer_id}")
def recommend(
    customer_id: str,
    top_k: int = 10
):

    user_embedding = user_model(
        tf.constant([customer_id])
    ).numpy().astype("float32")

    scores, indices = faiss_index.search(
        user_embedding,
        top_k
    )

    recommendations = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index != -1:

            recommendations.append({
                "article_id": str(article_ids[index]),
                "score": float(score)
            })

    return {
        "customer_id": customer_id,
        "recommendations": recommendations
    }