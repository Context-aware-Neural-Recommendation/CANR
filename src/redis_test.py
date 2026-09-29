import redis

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

print("Redis connection:", r.ping())

r.set("test_key", "Hello Redis")
print("Stored value:", r.get("test_key"))