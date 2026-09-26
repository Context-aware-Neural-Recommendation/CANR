import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRANSACTIONS_FILE = BASE_DIR / "data" / "features" / "transaction_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "features" / "training_data.csv"

CHUNK_SIZE = 500_000

print("Preparing training data...")

first_chunk = True
total_rows = 0

for chunk in pd.read_csv(TRANSACTIONS_FILE, chunksize=CHUNK_SIZE):

    # Keep only columns required for recommendation model
    chunk = chunk[
        [
            "customer_id",
            "article_id",
            "price",
            "year",
            "month",
            "day_of_week",
        ]
    ]

    # Remove invalid records
    chunk = chunk.dropna(
        subset=["customer_id", "article_id"]
    )

    # Write incrementally
    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )

    total_rows += len(chunk)
    first_chunk = False

    print(f"Processed {total_rows:,} rows...")

print()
print("Training data preparation completed!")
print(f"Total training rows: {total_rows:,}")
print(f"Saved to: {OUTPUT_FILE}")
