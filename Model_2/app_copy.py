import sys
print("Python executable:", sys.executable)

import os
import re
import pandas as pd
import numpy as np
import torch
from datasets import load_from_disk
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModel
from symspellpy.symspellpy import SymSpell, Verbosity

# Import only the functions from precompute_dataset.py
from precompute_dataset import cls_pooling, get_embeddings

# Check if faiss is available
try:
    import faiss
    print("Faiss version:", faiss.__version__)
except ImportError:
    print("Warning: Faiss not found. Install with: pip install faiss-cpu")

# === FastAPI App ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Load SymSpell ===
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# === Load Models (only for query embedding) ===
device = torch.device("cpu")
model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint).to(device)

# === Helper functions (using imported functions) ===
def embed_text(texts):
    """Get embeddings for search queries using the same functions as precompute_dataset.py"""
    print("→ Getting query embeddings...")
    embeddings_tensor = get_embeddings(texts, tokenizer, model, device)
    return embeddings_tensor.detach().cpu().numpy()

def correct_query(text):
    suggestion = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestion[0].term if suggestion else text

def extract_price(text):
    match = re.search(r"\b(\d{2,5})\b", text)
    return int(match.group(1)) if match else None

# === Define global dataset variable ===
dataset = None

def load_and_index_dataset():
    global dataset
    print("Loading precomputed dataset from disk...")
    dataset = load_from_disk("precomputed_dataset")
    dataset.load_faiss_index("embeddings", "faiss.index")
    print("✅ Dataset loaded.")

# === Routes ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/search")
def search_api(q: str):
    if dataset is None:
        return {"error": "Dataset not loaded. Please restart server."}

    corrected_query = correct_query(q)
    target_price = extract_price(corrected_query)
    
    # Get query embedding using the same functions as precompute_dataset.py
    query_embedding = embed_text([corrected_query])[0]

    scores, samples = dataset.get_nearest_examples(
        index_name="embeddings", query=query_embedding, k=20
    )

    results = pd.DataFrame(samples)
    results["similarity_score"] = scores

    keyword = corrected_query.lower().strip()
    if len(keyword.split()) == 1:
        boost_amount = 0.1
        boosted_scores = []
        for i, text in enumerate(results["text"]):
            score = results["similarity_score"].iloc[i]
            if keyword in text.lower():
                score += boost_amount
            boosted_scores.append(score)
        results["boosted_score"] = boosted_scores
        results = results.sort_values(by="boosted_score", ascending=False)
    else:
        results["boosted_score"] = results["similarity_score"]

    if target_price:
        def is_price_match(p):
            try:
                return abs(float(p) - target_price) <= 5
            except:
                return False
        results = results[results["Price"].apply(is_price_match)]

    results = results.head(10)
    return results.to_dict(orient="records")

@app.on_event("startup")
def startup_event():
    print("🔁 Loading precomputed dataset...")
    load_and_index_dataset()
    print("✅ Dataset ready.")
