import pandas as pd
import random
from pathlib import Path


TRAINING_FILE = Path("data/features/training_data.csv")
OUTPUT_FILE = Path("data/features/negative_sampling_data.csv")

NEGATIVES_PER_POSITIVE = 1
CHUNK_SIZE = 1_000


def create_negative_samples():

    print("Starting negative sampling...")

    # First collect unique article IDs
    print("Loading article IDs...")

    article_ids = set()

    for chunk in pd.read_csv(
        TRAINING_FILE,
        usecols=["article_id"],
        chunksize=CHUNK_SIZE
    ):
        article_ids.update(chunk["article_id"].dropna().astype(str).unique())

    article_ids = list(article_ids)

    print(f"Unique articles found: {len(article_ids):,}")

    first_chunk = True
    total_positive = 0
    total_negative = 0

    for chunk in pd.read_csv(
        TRAINING_FILE,
        usecols=["customer_id", "article_id"],
        chunksize=CHUNK_SIZE
    ):

        chunk["customer_id"] = chunk["customer_id"].astype(str)
        chunk["article_id"] = chunk["article_id"].astype(str)

        positive_data = chunk.copy()
        positive_data["label"] = 1

        negative_rows = []

        for _, row in chunk.iterrows():

            user_id = row["customer_id"]
            positive_item = row["article_id"]

            negative_item = random.choice(article_ids)

            while negative_item == positive_item:
                negative_item = random.choice(article_ids)

            negative_rows.append({
                "customer_id": user_id,
                "article_id": negative_item,
                "label": 0
            })

        negative_data = pd.DataFrame(negative_rows)

        output_data = pd.concat(
            [positive_data, negative_data],
            ignore_index=True
        )

        output_data.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_positive += len(positive_data)
        total_negative += len(negative_data)

        print(
            f"Processed: {total_positive:,} positive | "
            f"{total_negative:,} negative"
        )

    print("\nNegative sampling completed!")
    print(f"Positive samples: {total_positive:,}")
    print(f"Negative samples: {total_negative:,}")
    print(f"Total samples: {total_positive + total_negative:,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_negative_samples()