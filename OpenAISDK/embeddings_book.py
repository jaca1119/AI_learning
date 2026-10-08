import json
import math
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_DIR / "resources"
embeddings_model = "text-embedding-3-small"
BOOK_PATH = RESOURCES_DIR / "katarynka.txt"
INDEX_PATH = RESOURCES_DIR / f"{BOOK_PATH.stem}_index.json"

client = OpenAI()

def create_embeddings():
    book_text = BOOK_PATH.read_text(encoding="utf-8")
    chunks = []
    start = 0
    size = 1500
    overlap = 250

    while start < len(book_text):
        end = start + size
        chunk = book_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    response = client.embeddings.create(
        model=embeddings_model,
        input=chunks
    )

    index = {
        "embedding_model": embeddings_model,
        "source_file": BOOK_PATH.name,
        "chunks": [
            {
                "id": i,
                "text": chunk,
                "embedding": item.embedding
            }
            for i, (chunk, item) in enumerate(zip(chunks, response.data))
        ]
    }

    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def search_index(index, search: str):
    search_embedding_response = client.embeddings.create(
        model=embeddings_model,
        input=search
    )
    search_embedding = search_embedding_response.data[0].embedding

    scored_chunks =[
        {
        **chunk,
        "score": cosine_similarity(search_embedding, chunk["embedding"])
        } for chunk in index["chunks"]
    ]

    #return top few
    return sorted(scored_chunks, key=lambda item: item["score"], reverse=True)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    #a * b / (|a| * |b|)
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (length_a * length_b)



if __name__ == "__main__":
    if not INDEX_PATH.exists():
        create_embeddings()

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    question = "Jakie było zdanie pana Tomasza na temat katarynek?"
    found = search_index(index, question)
    for chunk in found[:3]:
        print("---------------------")
        print(chunk["text"])
        print("---------------------")

    question = "Dlaczego pan Tomasz zmienił zdanie o katarynkach?"
    found = search_index(index, question)
    for chunk in found[:3]:
        print("---------------------")
        print(chunk["text"])
        print("---------------------")
