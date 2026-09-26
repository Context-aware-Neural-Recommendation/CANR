import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("data/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def create_customer_features():
    print("Creating customer features...")

    customers = pd.read_csv(
        PROCESSED_DIR / "customers_processed.csv"
    )

    customer_features = customers[
        [
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency"
        ]
    ].copy()

    customer_features["age"] = pd.to_numeric(
        customer_features["age"],
        errors="coerce"
    )

    customer_features["age"] = customer_features["age"].fillna(
        customer_features["age"].median()
    )

    customer_features.to_csv(
        FEATURE_DIR / "customer_features.csv",
        index=False
    )

    print(
        f"Customer features created: "
        f"{len(customer_features):,} rows"
    )


def create_article_features():
    print("Creating article features...")

    articles = pd.read_csv(
        PROCESSED_DIR / "articles_processed.csv"
    )

    article_features = articles.copy()

    article_features.to_csv(
        FEATURE_DIR / "article_features.csv",
        index=False
    )

    print(
        f"Article features created: "
        f"{len(article_features):,} rows"
    )


def create_transaction_features():
    print("Creating transaction features...")

    input_file = PROCESSED_DIR / "transactions_processed.csv"
    output_file = FEATURE_DIR / "transaction_features.csv"

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_file,
        chunksize=100_000
    ):
        chunk["price"] = pd.to_numeric(
            chunk["price"],
            errors="coerce"
        )

        chunk["price"] = chunk["price"].fillna(0)

        chunk["sales_value"] = chunk["price"]

        chunk["transaction_date"] = pd.to_datetime(
            chunk["t_dat"],
            errors="coerce"
        )

        chunk["year"] = chunk["transaction_date"].dt.year
        chunk["month"] = chunk["transaction_date"].dt.month
        chunk["day_of_week"] = (
            chunk["transaction_date"].dt.dayofweek
        )

        output_columns = [
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_value",
            "year",
            "month",
            "day_of_week"
        ]

        chunk[output_columns].to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_rows += len(chunk)
        first_chunk = False

        print(
            f"Processed {total_rows:,} transactions..."
        )

    print(
        f"Transaction features created: "
        f"{total_rows:,} rows"
    )


if __name__ == "__main__":
    create_customer_features()
    create_article_features()
    create_transaction_features()

    print("\nFeature engineering completed successfully!")