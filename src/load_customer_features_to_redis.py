import csv
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
CSV_FILE = "Data/features/customer_features.csv"

client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

count = 0

with open(CSV_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        customer_id = row["customer_id"]

        profile = {
            "age": row["age"],
            "club_member_status": row["club_member_status"],
            "fashion_news_frequency": row["fashion_news_frequency"]
        }

        client.hset(
            f"user:{customer_id}",
            mapping=profile
        )

        count += 1

        if count % 10000 == 0:
            print(f"Loaded {count} customer profiles...")

print(f"Completed. Total customer profiles loaded: {count}")
