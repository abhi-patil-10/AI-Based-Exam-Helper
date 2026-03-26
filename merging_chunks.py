import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# ==============================
# Step 1 — Load Chunk Files
# ==============================

print("Loading chunk files...")

with open("../Project/combined_questions_chunks.json", "r", encoding="utf-8") as f:
    question_chunks = json.load(f)

with open("../Project/combined_syllabus_chunks.json", "r", encoding="utf-8") as f:
    syllabus_chunks = json.load(f)

# Merge both
all_chunks = question_chunks + syllabus_chunks

print(f"Question chunks: {len(question_chunks)}")
print(f"Syllabus chunks: {len(syllabus_chunks)}")
print(f"Total chunks: {len(all_chunks)}")

# ==============================
# Step 2 — Extract Text
# ==============================

texts = [chunk["text"] for chunk in all_chunks]

print("Total texts to embed:", len(texts))

# ==============================
# Step 3 — Load BGE-M3 Model
# ==============================

print("Loading embedding model (BGE-M3-small)...")

model = SentenceTransformer(
   "BAAI/bge-small-en",
    device="cpu"  # change to "cuda" if GPU available
)

# ==============================
# Step 4 — Create Embeddings
# ==============================

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings)

print("Embedding shape:", embeddings.shape)

# ==============================
# Step 5 — Create FAISS Index
# ==============================

dimension = embeddings.shape[1]

print("Creating FAISS index...")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Total vectors stored:", index.ntotal)

# ==============================
# Step 6 — Save FAISS Index
# ==============================

print("Saving FAISS index...")

faiss.write_index(
    index,
    "vector_index.faiss"
)

# ==============================
# Step 7 — Save Metadata
# ==============================

print("Saving metadata...")

with open("all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4)

print("✅ Vector DB created successfully!")
print("Files saved:")
print(" - vector_index.faiss")
print(" - all_chunks.json")