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
# Step 2 — Load All Chunks
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

def filtered_search(query, k=20):

    # 🔥 BGE works better with instruction prefix
    query_for_embedding = (
        "Represent this question for searching: "
        + query
    )

    # Create embedding
    query_embedding = np.array(
        model.encode([query_for_embedding])
    ).astype("float32")

    # 🔥 Search more candidates (IMPORTANT FIX)
    D, I = index.search(query_embedding, k * 10)

    results = []

    query_lower = query.lower()

    want_imp = "imp" in query_lower
    want_pyq = "pyq" in query_lower
    want_syllabus = "syllabus" in query_lower

    for idx in I[0]:

        chunk = all_chunks[idx]

        meta = chunk["metadata"]

        # 🚫 Skip syllabus unless asked
        if not want_syllabus:
            if meta["type"] == "SYLLABUS":
                continue

        # Filter IMP
        if want_imp and meta["type"] != "IMP":
            continue

        # Filter PYQ
        if want_pyq and meta["type"] != "PYQ":
            continue

        results.append(chunk)

        if len(results) == k:
            break

    return results


# ==============================
# Step 5 — Interactive Loop
# ==============================

if __name__ == "__main__":

    while True:

        user_query = input(
            "\nAsk something (or type exit): "
        )

        if user_query.lower() == "exit":
            break

        results = filtered_search(
            user_query,
            k=20
        )

        print("\nResults:\n")

        for r in results:

            print("-" * 40)

            print("Text:")
            print(r["text"])

            print("\nMetadata:")
            print(r["metadata"])