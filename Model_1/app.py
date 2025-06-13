from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset and pre-built FAISS index
with open("fikse_data.pkl", "rb") as f:
    DATA = pickle.load(f)

index = faiss.read_index("fikse_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query: str, top_k: int = 5):
    # Encode the query using the sentence transformer model
    query_embedding = model.encode([query]).astype(np.float32)

    # Use the loaded FAISS index to search for top_k nearest neighbors
    D, I = index.search(query_embedding, top_k)

    results = []
    for i, dist in zip(I[0], D[0]):
        item = DATA[i]
        item["score"] = float(dist)
        results.append(item)
    return results

@app.get("/")
def home():
    return {"message": "Welcome to FikseSearch API"}

@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Your search query"),
    top_k: int = Query(5, description="Number of results to return"),
):
    results = search(q, top_k=top_k)
    return {"query": q, "results": results}
