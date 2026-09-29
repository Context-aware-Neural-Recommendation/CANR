import redis
import faiss
import numpy as np

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

index = faiss.read_index("models/faiss/article_index.faiss")

article_ids = np.load(
    "models/faiss/article_ids.npy",
    allow_pickle=True
)

embeddings = np.load(
    "models/embeddings/item_embeddings.npy"
)

query_index = 0

query_vector = embeddings[query_index].astype(
    "float32"
).reshape(1, -1)

distances, positions = index.search(
    query_vector,
    5
)

print("Recommended Articles:")

for i, position in enumerate(positions[0]):
    article_id = article_ids[position]

    exists = redis_client.exists(
        f"item:{article_id}"
    )

    print(
        f"{i + 1}. Article ID: {article_id} | "
        f"Distance: {distances[0][i]:.4f} | "
        f"Redis: {'Yes' if exists else 'No'}"
    )
