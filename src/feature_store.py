import redis


class FeatureStore:

    def __init__(self):
        self.client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )

    def set_user_profile(self, customer_id, profile):
        self.client.hset(
            f"user:{customer_id}",
            mapping=profile
        )

    def get_user_profile(self, customer_id):
        return self.client.hgetall(
            f"user:{customer_id}"
        )

    def set_item_vector(self, article_id, vector):
        self.client.hset(
            f"item:{article_id}",
            "vector",
            ",".join(map(str, vector))
        )

    def get_item_vector(self, article_id):
        value = self.client.hget(
            f"item:{article_id}",
            "vector"
        )

        if value is None:
            return None

        return [float(x) for x in value.split(",")]


if __name__ == "__main__":

    store = FeatureStore()

    store.set_user_profile(
        "00000c5e3a",
        {
            "age": "25",
            "gender": "F"
        }
    )

    print("User Profile:")
    print(store.get_user_profile("00000c5e3a"))

    test_vector = [0.1, 0.2, 0.3, 0.4]

    store.set_item_vector(
        "0108775044",
        test_vector
    )

    print("\nItem Vector:")
    print(store.get_item_vector("0108775044"))