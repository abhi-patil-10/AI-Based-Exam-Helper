import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# ==============================
# Step 1 — Load FAISS Index
# ==============================

print("Loading FAISS index...")

index = faiss.read_index("vector_index.faiss")


# ==============================
# Step 2 — Load Metadata
# ==============================

print("Loading chunks metadata...")

with open("all_chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)


# ==============================
# Step 3 — Load BGE-M3 Model
# ==============================

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu"
)


# ==============================
# Step 4 — Search Function
# ==============================

def search_query(query, top_k=10):

    print("\nUser Query:", query)

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding)

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("\nTop Results:\n")

    results = []

    for i in indices[0]:

        chunk = all_chunks[i]

        results.append(chunk)

        print(chunk["text"])
        print("-" * 50)

    return results


# ==============================
# Step 5 — Test Query
# ==============================

if __name__ == "__main__":

    while True:

        user_query = input("\nAsk something (or type exit): ")

        if user_query.lower() == "exit":
            break

        search_query(user_query)