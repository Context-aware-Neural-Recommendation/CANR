import redis

client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

user_id = "user_001"

user_profile = {
    "age": "24",
    "gender": "F",
    "preferred_category": "Dresses",
    "preferred_color": "Black"
}

client.hset(
    f"user:{user_id}",
    mapping=user_profile
)

profile = client.hgetall(f"user:{user_id}")

print("User profile stored successfully!")
print("Profile:", profile)
