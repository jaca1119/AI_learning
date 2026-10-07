import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
embeddings_model = "text-embedding-3-small"

client = OpenAI()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    #a * b / (|a| * |b|)
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (length_a * length_b)

if __name__ == "__main__":
    search = "car needs gas"
    text = [
        "She left her car keys on the kitchen table.",
        "Gas prices increased across Europe last month.",
        "The company needs a new marketing strategy.",
        "The puppy is asleep on the couch.",
        "The vehicle requires fuel before continuing the journey."
    ]

    text_embeddings = client.embeddings.create(
        model=embeddings_model,
        input=text
    ).data

    query_embeddings = client.embeddings.create(
        model=embeddings_model,
        input=search
    ).data[0].embedding

    for text, item in zip(text, text_embeddings):
        similarity = cosine_similarity(query_embeddings, item.embedding)
        print(f"{similarity} - {text}")