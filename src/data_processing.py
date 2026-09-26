import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_articles():
    print("Processing articles.csv...")

    articles = pd.read_csv(RAW_DIR / "articles.csv")

    articles.to_csv(
        PROCESSED_DIR / "articles_processed.csv",
        index=False
    )

    print(f"Articles processed: {len(articles):,} rows")


def process_customers():
    print("Processing customers.csv...")

    customers = pd.read_csv(RAW_DIR / "customers.csv")

    customers.to_csv(
        PROCESSED_DIR / "customers_processed.csv",
        index=False
    )

    print(f"Customers processed: {len(customers):,} rows")


def process_transactions():
    print("Processing transactions_train.csv...")

    output_file = PROCESSED_DIR / "transactions_processed.csv"

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        RAW_DIR / "transactions_train.csv",
        chunksize=100_000
    ):
        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_rows += len(chunk)
        first_chunk = False

        print(f"Processed {total_rows:,} transactions...")


if __name__ == "__main__":
    process_articles()
    process_customers()
    process_transactions()

    print("\nData processing completed successfully!")