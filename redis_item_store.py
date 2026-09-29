import redis
import numpy as np

client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

embeddings = np.load("models/embeddings/item_embeddings.npy")
article_ids = np.load("models/embeddings/article_ids.npy")

print("Loading item embeddings...")
print("Total items:", len(article_ids))

for article_id, vector in zip(article_ids, embeddings):
    vector_string = ",".join(map(str, vector))
    
    client.set(
        f"item:{article_id}",
        vector_string
    )

print("All item embeddings stored in Redis successfully!")
