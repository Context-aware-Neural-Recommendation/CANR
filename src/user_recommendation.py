import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import faiss
import numpy as np
import redis

# -----------------------------------------
# Paths
# -----------------------------------------

USER_MODEL = "models/exported/user_model.keras"
FAISS_INDEX = "models/faiss/article_index.faiss"
ARTICLE_IDS = "models/faiss/article_ids.npy"
VOCABULARY = "data/features/vocabularies.json"


# -----------------------------------------
# Load User Tower
# -----------------------------------------

print("Loading User Tower...")

user_model = tf.keras.models.load_model(
    USER_MODEL,
    compile=False
)

print("User Tower loaded successfully!")


# -----------------------------------------
# Load FAISS index
# -----------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(FAISS_INDEX)

article_ids = np.load(
    ARTICLE_IDS,
    allow_pickle=True
)

print("FAISS index loaded successfully!")
print("Index vectors:", index.ntotal)


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

print("Redis connected:", redis_client.ping())

# -----------------------------------------
# Select customer
# -----------------------------------------

import json

with open(
    VOCABULARY,
    "r",
    encoding="utf-8"
) as file:
    vocabularies = json.load(file)

import sys

customer_id = str(vocabularies["customer_id"][0])

if len(sys.argv) > 1:
    customer_id = str(sys.argv[1])

print("Customer ID:", customer_id)


# -----------------------------------------
# Generate User Embedding
# -----------------------------------------

user_embedding = user_model(
    tf.constant([customer_id])
)

user_embedding = user_embedding.numpy().astype(
    "float32"
)


print(
    "User embedding shape:",
    user_embedding.shape
)


# -----------------------------------------
# FAISS Retrieval
# -----------------------------------------

TOP_K = 5

distances, positions = index.search(
    user_embedding,
    TOP_K
)


# -----------------------------------------
# Display Recommendations
# -----------------------------------------

print("\nRecommended Articles:")

for i, position in enumerate(positions[0]):

    article_id = article_ids[position]

    redis_key = f"item:{article_id}"

    exists = redis_client.exists(redis_key)

    print(
        f"{i + 1}. Article ID: {article_id} | "
        f"Distance: {distances[0][i]:.4f} | "
        f"Redis: {'Yes' if exists else 'No'}"
    )