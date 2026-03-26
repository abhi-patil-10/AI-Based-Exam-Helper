# ==============================
# STEP 1 — Import Libraries
# ==============================

from flask import Flask, request, render_template
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ==============================
# STEP 2 — Create Flask App
# ==============================

print("STEP 2: Creating Flask App...")
app = Flask(__name__)

# ==============================
# STEP 3 — Load FAISS Index
# ==============================

print("STEP 3: Loading FAISS index...")

index = faiss.read_index("vector_index.faiss")

print("STEP 3 DONE: FAISS index loaded")

# ==============================
# STEP 4 — Load Chunks JSON
# ==============================

print("STEP 4: Loading chunks JSON...")

with open("all_chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

print("STEP 4 DONE: Chunks loaded")

# ==============================
# STEP 5 — Load Embedding Model
# ==============================

print("STEP 5: Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en",
    device="cpu"
)

print("STEP 5 DONE: Model loaded")

# ==============================
# STEP 6 — Model Warmup
# ==============================

print("STEP 6: Warming up model...")

model.encode(["test"])

print("STEP 6 DONE: Model ready")

print("STEP 7: System Ready!")

# ==============================
# STEP 8 — LLM Inference Function
# ==============================

def inference(prompt):

    print("STEP 8: Sending prompt to LLM...")

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:latest",
            "prompt": prompt,
            "stream": False
        }
    )

    print("STEP 8 DONE: Response received")

    return r.json()["response"]

# ==============================
# STEP 9 — Search Function
# ==============================

def filtered_search(query, k=8):

    print("STEP 9: Starting FAISS search...")

    query_for_embedding = (
        "Represent this question for searching: "
        + query
    )

    # STEP 9.1 — Generate embedding
    print("STEP 9.1: Generating embedding...")

    query_embedding = np.array(
        model.encode([query_for_embedding])
    ).astype("float32")

    # STEP 9.2 — Search FAISS
    print("STEP 9.2: Searching index...")

    D, I = index.search(query_embedding, k * 3)

    results = []

    # STEP 9.3 — Collect Results
    print("STEP 9.3: Collecting results...")

    for idx in I[0]:

        chunk = all_chunks[idx]
        results.append(chunk)

        if len(results) == k:
            break

    print("STEP 9 DONE: Search completed")

    return results

# ==============================
# STEP 10 — Build Prompt
# ==============================

def build_prompt(user_query, results):

    print("STEP 10: Building prompt...")

    context_text = "\n\n".join(
        chunk["text"]
        for chunk in results
    )

    system_prompt = """
    (Your full system prompt here — unchanged)
    """

    prompt = f"""
    SYSTEM:
    {system_prompt}

    -----------------------------------

    RAG CONTEXT:
    {context_text}

    -----------------------------------

    USER QUESTION:
    {user_query}

    -----------------------------------

    Generate the final answer.
    """

    print("STEP 10 DONE: Prompt ready")

    return prompt

# ==============================
# STEP 11 — Home Route
# ==============================

@app.route("/", methods=['GET', 'POST'])
def home():

    print("STEP 11: Home route accessed")

    response = None

    if request.method == 'POST':

        print("STEP 11.1: POST request received")

        query = request.form['query']

        print("STEP 11.2: User Query:", query)

        # STEP 11.3 — Search
        results = filtered_search(query)

        # STEP 11.4 — Build Prompt
        prompt = build_prompt(
            query,
            results
        )

        # STEP 11.5 — LLM Response
        response = inference(prompt)

        print("STEP 11.6: LLM Response received")

    print("STEP 11 DONE: Returning template")

    return render_template(
        "home.html",
        response=response
    )

# ==============================
# STEP 12 — Other Routes
# ==============================

@app.route("/about")
def about():

    print("STEP 12: About page opened")

    return render_template("about.html")

@app.route("/contact")
def contact():

    print("STEP 12: Contact page opened")

    return render_template("contact.html")

# ================================
# STEP 13 — Run Flask Server
# ================================

if __name__ == "__main__":

    print("STEP 13: Starting Flask server...")

    app.run(debug=True)