import json
import numpy as np
import faiss
from model import embed

# Load dummy data
with open("data.json") as f:
    DATA = json.load(f)

# Prepare text for embedding - use description only or combine fields if needed
def format_record(record):
    return record['description']

# Create embeddings (and convert to float32)
EMBEDDINGS = [embed(format_record(rec)).astype('float32') for rec in DATA]

# Normalize embeddings for cosine similarity search
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

embedding_matrix = np.array(EMBEDDINGS)
embedding_matrix = normalize_vectors(embedding_matrix)

# Build FAISS index for cosine similarity (use IndexFlatIP on normalized vectors)
dim = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity on normalized vectors
index.add(embedding_matrix)

def search(query, k=1):
    query_vec = embed(query).astype('float32')
    query_vec = query_vec / np.linalg.norm(query_vec)  # normalize query vector
    
    D, I = index.search(np.array([query_vec]), k)
    results = []
    for i, score in zip(I[0], D[0]):
        rec = DATA[i].copy()
        rec['score'] = float(score)  # Add similarity score for debugging or frontend use
        results.append(rec)
    return results
