import json
import tensorflow as tf

from query_tower import QueryTower
from candidate_tower import CandidateTower


class TwoTowerModel(tf.keras.Model):

    def __init__(self, customer_ids, article_ids, embedding_dim=32):
        super().__init__()

        self.query_tower = QueryTower(
            customer_ids,
            embedding_dim=embedding_dim
        )

        self.candidate_tower = CandidateTower(
            article_ids,
            embedding_dim=embedding_dim
        )

    def call(self, customer_id, article_id):

        query_embedding = self.query_tower(customer_id)
        candidate_embedding = self.candidate_tower(article_id)

        return query_embedding, candidate_embedding


if __name__ == "__main__":

    with open(
        "data/features/vocabularies.json",
        "r",
        encoding="utf-8"
    ) as file:
        vocabularies = json.load(file)

    customer_ids = vocabularies["customer_id"]
    article_ids = vocabularies["article_id"]

    model = TwoTowerModel(
        customer_ids,
        article_ids
    )

    sample_customer = tf.constant([customer_ids[0]])
    sample_article = tf.constant([article_ids[0]])

    query_embedding, candidate_embedding = model(
        sample_customer,
        sample_article
    )

    similarity = tf.reduce_sum(
        query_embedding * candidate_embedding,
        axis=1
    )

    print("\nTwo-Tower Neural Network test successful!")
    print("User embedding shape:", query_embedding.shape)
    print("Item embedding shape:", candidate_embedding.shape)
    print("Similarity score:", similarity.numpy())