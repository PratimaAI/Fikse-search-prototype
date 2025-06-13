from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    vec = model.encode([text])[0]
    return np.array(vec, dtype=np.float32)
