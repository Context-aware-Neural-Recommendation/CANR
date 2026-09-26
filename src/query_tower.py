import tensorflow as tf


class QueryTower(tf.keras.Model):

    def __init__(self, customer_ids, embedding_dim=32):
        super().__init__()

        self.customer_lookup = tf.keras.layers.StringLookup(
            vocabulary=customer_ids,
            mask_token=None
        )

        self.customer_embedding = tf.keras.layers.Embedding(
            input_dim=self.customer_lookup.vocabulary_size(),
            output_dim=embedding_dim
        )

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32)
        ])

    def call(self, customer_id):

        customer_id = self.customer_lookup(customer_id)

        embedding = self.customer_embedding(customer_id)

        return self.dense(embedding)

if __name__ == "__main__":
    import json

    with open(
        "data/features/vocabularies.json",
        "r",
        encoding="utf-8"
    ) as file:
        vocabularies = json.load(file)

    customer_ids = vocabularies["customer_id"]

    model = QueryTower(customer_ids)

    sample_user = tf.constant([customer_ids[0]])

    output = model(sample_user)

    print("\nQuery Tower test successful!")
    print("Sample customer ID:", customer_ids[0])
    print("Embedding shape:", output.shape)