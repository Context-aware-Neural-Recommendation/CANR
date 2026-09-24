from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    min,
    max,
    sum as spark_sum
)


def main():

    # --------------------------------------------------
    # 1. Create Spark session
    # --------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("H&M Transaction Inspection")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "=" * 60)
    print("H&M TRANSACTION DATASET INSPECTION")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Read transactions
    # --------------------------------------------------

    transaction_path = "data/raw/transactions_train.csv"

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(transaction_path)
    )

    # --------------------------------------------------
    # 3. Display schema
    # --------------------------------------------------

    print("\n--- SCHEMA ---")
    transactions.printSchema()

    # --------------------------------------------------
    # 4. Number of rows
    # --------------------------------------------------

    print("\n--- ROW COUNT ---")

    total_rows = transactions.count()

    print(f"Total transactions: {total_rows:,}")

    # --------------------------------------------------
    # 5. Preview data
    # --------------------------------------------------

    print("\n--- SAMPLE RECORDS ---")

    transactions.show(5, truncate=False)

        # --------------------------------------------------
    # 6. Missing values
    # --------------------------------------------------

    print("\n--- MISSING VALUES ---")

    missing_values = transactions.select(
        [
            (
                count("*") - count(col(column_name))
            ).alias(column_name)
            for column_name in transactions.columns
        ]
    )

    missing_values.show()

    # --------------------------------------------------
    # 7. Unique customers
    # --------------------------------------------------

    print("\n--- UNIQUE CUSTOMERS ---")

    unique_customers = transactions.select(
        countDistinct("customer_id").alias("unique_customers")
    ).collect()[0]["unique_customers"]

    print(f"Unique customers: {unique_customers:,}")

    # --------------------------------------------------
    # 8. Unique articles
    # --------------------------------------------------

    print("\n--- UNIQUE ARTICLES ---")

    unique_articles = transactions.select(
        countDistinct("article_id").alias("unique_articles")
    ).collect()[0]["unique_articles"]

    print(f"Unique articles: {unique_articles:,}")

    # --------------------------------------------------
    # 9. Date range
    # --------------------------------------------------

    print("\n--- DATE RANGE ---")

    date_range = transactions.select(
        min("t_dat").alias("start_date"),
        max("t_dat").alias("end_date")
    ).collect()[0]

    print(f"Start date: {date_range['start_date']}")
    print(f"End date:   {date_range['end_date']}")

    # --------------------------------------------------
    # 10. Price statistics
    # --------------------------------------------------

    print("\n--- PRICE STATISTICS ---")

    transactions.select(
        "price"
    ).summary(
        "count",
        "mean",
        "stddev",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ).show()

    # --------------------------------------------------
    # 11. Sales channel distribution
    # --------------------------------------------------

    print("\n--- SALES CHANNEL DISTRIBUTION ---")

    transactions.groupBy(
        "sales_channel_id"
    ).count().orderBy(
        col("count").desc()
    ).show()

    # --------------------------------------------------
    # 12. Stop Spark
    # --------------------------------------------------

    spark.stop()

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()