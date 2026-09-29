import os
import numpy as np
import redis


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "embeddings",
    "item_embeddings.npy"
)

ARTICLE_IDS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "embeddings",
    "article_ids.npy"
)


client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)


print("Loading item embeddings...")

embeddings = np.load(EMBEDDINGS_PATH)
article_ids = np.load(
    ARTICLE_IDS_PATH,
    allow_pickle=True
)

print("Embeddings shape:", embeddings.shape)
print("Article IDs:", article_ids.shape)

print("\nStoring item vectors in Redis...")

for article_id, vector in zip(article_ids, embeddings):

    client.hset(
        f"item:{article_id}",
        "vector",
        ",".join(map(str, vector))
    )

print("\nItem embeddings stored successfully!")
print("Total items stored:", len(article_ids))