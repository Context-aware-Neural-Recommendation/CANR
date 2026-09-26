import pandas as pd
import json
from pathlib import Path


FEATURE_DIR = Path("data/features")


def create_vocabularies():
    print("Creating user and item vocabularies...")

    customers = pd.read_csv(
        FEATURE_DIR / "customers_cold_start.csv",
        usecols=["customer_id"]
    )

    articles = pd.read_csv(
        FEATURE_DIR / "articles_cold_start.csv"
    )

    vocabularies = {}

    # User ID vocabulary
    vocabularies["customer_id"] = (
        customers["customer_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Item ID vocabulary
    vocabularies["article_id"] = (
        articles["article_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Item categorical vocabularies
    categorical_columns = [
        "product_code",
        "prod_name",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "department_name",
        "index_name",
        "section_name",
        "garment_group_name"
    ]

    for column in categorical_columns:
        if column in articles.columns:
            vocabularies[column] = (
                articles[column]
                .fillna("UNKNOWN")
                .astype(str)
                .unique()
                .tolist()
            )

    output_file = FEATURE_DIR / "vocabularies.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            vocabularies,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nVocabulary creation completed successfully!")
    print(f"Users: {len(vocabularies['customer_id']):,}")
    print(f"Items: {len(vocabularies['article_id']):,}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    create_vocabularies()