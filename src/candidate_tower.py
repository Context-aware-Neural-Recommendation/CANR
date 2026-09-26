import tensorflow as tf


class CandidateTower(tf.keras.Model):

    def __init__(self, article_ids, embedding_dim=32):
        super().__init__()

        self.article_lookup = tf.keras.layers.StringLookup(
            vocabulary=article_ids,
            mask_token=None
        )

        self.article_embedding = tf.keras.layers.Embedding(
            input_dim=self.article_lookup.vocabulary_size(),
            output_dim=embedding_dim
        )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32)
        ])

    def call(self, article_id):

        article_id = self.article_lookup(article_id)

        embedding = self.article_embedding(article_id)

        return self.dense(embedding)


if __name__ == "__main__":
    import json

    with open(
        "data/features/vocabularies.json",
        "r",
        encoding="utf-8"
    ) as file:
        vocabularies = json.load(file)

    article_ids = vocabularies["article_id"]

    model = CandidateTower(article_ids)

    sample_item = tf.constant([article_ids[0]])

    output = model(sample_item)

    print("\nCandidate Tower test successful!")
    print("Sample article ID:", article_ids[0])
    print("Embedding shape:", output.shape)