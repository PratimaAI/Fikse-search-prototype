import sys
print("Python executable:", sys.executable)

import os
import re
import faiss
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
from symspellpy.symspellpy import SymSpell, Verbosity
from contextlib import asynccontextmanager

print("Faiss version:", faiss.__version__)

# === FastAPI App ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Load SymSpell ===
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# === Load Models ===
device = torch.device("cpu")
model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint).to(device)

# === Define helper functions ===
def concatenate_text(batch):
    texts = []
    for i in range(len(batch["Type of Repairer"])):
        text = (
            str(batch["Type of Repairer"][i]) + "\n"
            + str(batch["Type of category"][i]) + "\n"
            + str(batch["Type of garment in category"][i]) + "\n"
            + str(batch["Service"][i]) + "\n"
            + str(batch["Description"][i]) + "\n"
            + str(batch["Price"][i]) + "\n"
            + str(batch["Estimated time in hours"][i])
        )
        texts.append(text)
    return {"text": texts}

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    print("→ Tokenizing...")
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    print("→ Running model inference...")
    with torch.no_grad():
        model_output = model(**encoded_input)
    print("→ Inference done.")
    return cls_pooling(model_output)


def embed_batch(batch):
    print("First text in batch:", batch["text"][0][:100])  # Preview first 100 chars
    embeddings_tensor = get_embeddings(batch["text"])
    embeddings_np = embeddings_tensor.detach().cpu().numpy()
    return {"embeddings": embeddings_np}

def embed_text(texts):
    embeddings_tensor = get_embeddings(texts)
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
    df = pd.read_csv("Dataset_categories.csv")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(concatenate_text, batched=True)
    dataset = dataset.map(embed_batch, batched=True, batch_size=1)
    dataset.add_faiss_index(column="embeddings")

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
async def startup_event():
    print("🔁 Running startup tasks...")
    load_and_index_dataset()
    print("✅ FAISS Index built at startup.")
