import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DATA = os.path.join(
    BASE_DIR,
    "data",
    "features",
    "training_data.csv"
)


# -----------------------------------------
# Load only a small sample for testing
# -----------------------------------------

print("Loading sample training data...")

df = pd.read_csv(
    TRAINING_DATA,
    nrows=1000000
)

print("Sample shape:", df.shape)


# -----------------------------------------
# Prepare IDs
# -----------------------------------------

df["customer_id"] = df["customer_id"].astype(str)
df["article_id"] = df["article_id"].astype(str)


# -----------------------------------------
# Create vocabularies
# -----------------------------------------

customer_ids = df["customer_id"].unique().tolist()
article_ids = df["article_id"].unique().tolist()

print("Unique customers:", len(customer_ids))
print("Unique articles:", len(article_ids))


# -----------------------------------------
# User Tower
# -----------------------------------------

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=customer_ids,
        mask_token=None
    ),
    tf.keras.layers.Embedding(
        len(customer_ids) + 1,
        64
    )
])


# -----------------------------------------
# Item Tower
# -----------------------------------------

item_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=article_ids,
        mask_token=None
    ),
    tf.keras.layers.Embedding(
        len(article_ids) + 1,
        64
    )
])


# -----------------------------------------
# Two-Tower Model
# -----------------------------------------

class TwoTowerModel(tfrs.models.Model):

    def __init__(self, user_model, item_model):
        super().__init__()

        self.user_model = user_model
        self.item_model = item_model

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(self, features, training=False):

        user_embeddings = self.user_model(
            features["customer_id"]
        )

        item_embeddings = self.item_model(
            features["article_id"]
        )

        return self.task(
            user_embeddings,
            item_embeddings
        )


# -----------------------------------------
# Create model
# -----------------------------------------

model = TwoTowerModel(
    user_model,
    item_model
)


# -----------------------------------------
# Compile
# -----------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=0.1
    )
)


# -----------------------------------------
# TensorFlow Dataset
# -----------------------------------------

dataset = tf.data.Dataset.from_tensor_slices({
    "customer_id": df["customer_id"].values,
    "article_id": df["article_id"].values
})


dataset = dataset.shuffle(
    buffer_size=100000
)

dataset = dataset.batch(2048)


# -----------------------------------------
# Test training
# -----------------------------------------

print("Starting test training...")

model.fit(
    dataset,
    epochs=1
)

print("Two-Tower model training completed successfully!")

MODEL_DIR = os.path.join(BASE_DIR, "models", "two_tower")

os.makedirs(MODEL_DIR, exist_ok=True)

model.user_model.save(
    os.path.join(MODEL_DIR, "user_model.keras")
)

model.item_model.save(
    os.path.join(MODEL_DIR, "item_model.keras")
)

print("User and item tower models saved successfully!")