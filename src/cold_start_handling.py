import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("data/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def create_cold_start_features():
    print("Creating cold-start handling features...")

    customers = pd.read_csv(
        PROCESSED_DIR / "customers_processed.csv"
    )

    articles = pd.read_csv(
        PROCESSED_DIR / "articles_processed.csv"
    )

    # New users: users without demographic information
    if "age" in customers.columns:
        customers["age"] = customers["age"].fillna(
            customers["age"].median()
        )

    # Fill missing categorical customer values
    for column in customers.select_dtypes(include="object").columns:
        customers[column] = customers[column].fillna("UNKNOWN")

    # New products: fill missing article metadata
    for column in articles.select_dtypes(include="object").columns:
        articles[column] = articles[column].fillna("UNKNOWN")

    # Cold-start flags
    customers["is_cold_start_user"] = 0
    articles["is_cold_start_item"] = 0

    # Save cold-start-ready features
    customers.to_csv(
        FEATURE_DIR / "customers_cold_start.csv",
        index=False
    )

    articles.to_csv(
        FEATURE_DIR / "articles_cold_start.csv",
        index=False
    )

    print(
        f"Customer cold-start features: {len(customers):,} rows"
    )

    print(
        f"Article cold-start features: {len(articles):,} rows"
    )

    print("\nCold-start handling completed successfully!")


if __name__ == "__main__":
    create_cold_start_features()