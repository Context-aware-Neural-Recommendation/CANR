import os
import numpy as np
import tensorflow as tf


MODEL_DIR = "models"
OUTPUT_DIR = "models/embeddings"

ITEM_MODEL_PATH = os.path.join(MODEL_DIR, "item_model.keras")
USER_MODEL_PATH = os.path.join(MODEL_DIR, "user_model.keras")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading item model...")
    item_model = tf.keras.models.load_model(ITEM_MODEL_PATH)

    print("Loading user model...")
    user_model = tf.keras.models.load_model(USER_MODEL_PATH)

    print("Item model loaded successfully.")
    print("User model loaded successfully.")

    print("\nModel information:")
    print(f"Item model output shape: {item_model.output_shape}")
    print(f"User model output shape: {user_model.output_shape}")

    print("\nEmbedding generation pipeline is ready.")
    print("Next step: generate embeddings using the trained vocabularies.")


if __name__ == "__main__":
    main()