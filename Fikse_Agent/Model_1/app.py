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

import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def lemmatize_and_lower(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])


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
    normalized_query = lemmatize_and_lower(corrected_query)
    target_price = extract_price(corrected_query)
    
    # STAGE 1: Get semantic candidates (optimized for repair services)
    print(f"🔍 Searching for: '{q}' (normalized: '{normalized_query}')")
    query_embedding = embed_text([normalized_query])[0]
    scores, samples = dataset.get_nearest_examples(
        index_name="embeddings", query=query_embedding, k=100  # More candidates for better filtering
    )
    
    all_results = pd.DataFrame(samples)
    all_results["similarity_score"] = scores
    
    # STAGE 2: Smart keyword matching optimized for repair services
    search_terms = [term.strip() for term in corrected_query.lower().split() if len(term.strip()) > 2]  # Skip short words
    
    # Categorize matches by relevance
    exact_service_matches = []    # Service field exact matches (highest priority)
    partial_service_matches = []  # Service field partial matches
    description_matches = []      # Description field matches
    general_matches = []          # Any field matches
    semantic_only = []           # No keyword matches
    
    print(f"🎯 Looking for terms: {search_terms}")
    
    for i, row in all_results.iterrows():
        service_lower = str(row["Service"]).lower()
        description_lower = str(row["Description"]).lower()
        garment_lower = str(row["Type of garment in category"]).lower()
        repairer_lower = str(row["Type of Repairer"]).lower()
        
        match_found = False
        match_details = []
        
        # Check each search term
        for term in search_terms:
            # Highest priority: Exact service match
            if term == service_lower:
                exact_service_matches.append((row, scores[i], f"exact_service:{term}"))
                match_found = True
                break
            # High priority: Partial service match
            elif term in service_lower:
                partial_service_matches.append((row, scores[i], f"partial_service:{term}"))
                match_found = True
                break
            # Medium priority: Description match
            elif term in description_lower:
                description_matches.append((row, scores[i], f"description:{term}"))
                match_found = True
                match_details.append(f"desc:{term}")
            # Lower priority: Garment/Repairer match
            elif term in garment_lower or term in repairer_lower:
                general_matches.append((row, scores[i], f"general:{term}"))
                match_found = True
                match_details.append(f"general:{term}")
        
        if not match_found:
            semantic_only.append((row, scores[i], "semantic_only"))
    
    print(f"📊 Match breakdown:")
    print(f"  - Exact service matches: {len(exact_service_matches)}")
    print(f"  - Partial service matches: {len(partial_service_matches)}")
    print(f"  - Description matches: {len(description_matches)}")
    print(f"  - General matches: {len(general_matches)}")
    print(f"  - Semantic only: {len(semantic_only)}")
    
    # STAGE 3: Combine results with smart prioritization
    final_results = []
    
    # Add matches in priority order
    all_match_groups = [
        (exact_service_matches, "exact_service"),
        (partial_service_matches, "partial_service"), 
        (description_matches, "description"),
        (general_matches, "general"),
        (semantic_only, "semantic")
    ]
    
    for match_group, match_type in all_match_groups:
        if len(final_results) >= 10:
            break
            
        # Sort each group by semantic similarity
        match_group.sort(key=lambda x: x[1], reverse=True)
        
        remaining_slots = 10 - len(final_results)
        for row, score, match_detail in match_group[:remaining_slots]:
            row_dict = row.to_dict()
            row_dict["similarity_score"] = float(score)  # Convert numpy.float32 to Python float
            row_dict["match_type"] = match_type
            row_dict["match_detail"] = match_detail
            row_dict["search_terms"] = search_terms
            final_results.append(row_dict)
    
    # Apply price filter if needed
    if target_price:
        def is_price_match(result):
            try:
                return abs(float(result["Price"]) - target_price) <= 50  # More flexible for repair prices
            except:
                return False
        final_results = [r for r in final_results if is_price_match(r)]
        print(f"💰 Price filter applied: {target_price} ± 50")
    
    print(f"🎯 Returning {len(final_results)} results")
    for i, result in enumerate(final_results[:5]):  # Show first 5 for debugging
        print(f"  {i+1}. {result['Service']} ({result['match_type']}) - Score: {result['similarity_score']:.2f}")
    
    return final_results[:10]

@app.get("/search_strategy")
def get_search_strategy():
    """Explain the two-stage search strategy being used"""
    return {
        "search_strategy": "Two-Stage Hybrid Search",
        "description": "Exact keyword matches first, semantic search second",
        "stages": [
            {
                "stage": 1,
                "name": "Semantic Candidate Retrieval",
                "description": "Use FAISS + embeddings to get 50 semantically similar candidates",
                "purpose": "Cast a wide net to find all potentially relevant results"
            },
            {
                "stage": 2,
                "name": "Exact Keyword Filtering", 
                "description": "Separate results that contain ALL search terms from those that don't",
                "purpose": "Prioritize exact matches that users are explicitly looking for"
            },
            {
                "stage": 3,
                "name": "Result Ranking",
                "description": "Show exact matches first, then semantic matches to fill remaining slots",
                "purpose": "Give users what they asked for, plus discovery of related items"
            }
        ],
        "benefits": [
            "Predictable: Users get exact matches first",
            "Fast: Simple logic, no complex scoring",
            "Discovery: Semantic search still helps find related items",
            "Scalable: Works with any dataset without configuration"
        ]
    }

@app.get("/debug_search")
def debug_search(q: str):
    """Debug endpoint to understand text processing and matching"""
    if dataset is None:
        return {"error": "Dataset not loaded. Please restart server."}

    corrected_query = correct_query(q)
    normalized_query = lemmatize_and_lower(corrected_query)
    
    print(f"Original query: '{q}'")
    print(f"Corrected query: '{corrected_query}'")
    print(f"Normalized query: '{normalized_query}'")
    
    # Get a few sample entries to show text processing
    query_embedding = embed_text([normalized_query])[0]
    scores, samples = dataset.get_nearest_examples(
        index_name="embeddings", query=query_embedding, k=5
    )
    
    results = pd.DataFrame(samples)
    debug_info = []
    
    for i in range(min(5, len(results))):
        row = results.iloc[i]
        debug_entry = {
            "service": row["Service"],
            "service_lower": str(row["Service"]).lower(),
            "service_lemmatized": lemmatize_and_lower(str(row["Service"])),
            "description": row["Description"],
            "text_field": row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"],
            "similarity_score": scores[i]
        }
        debug_info.append(debug_entry)
    
    return {
        "query_processing": {
            "original": q,
            "corrected": corrected_query,
            "normalized": normalized_query
        },
        "sample_entries": debug_info
    }

@app.on_event("startup")
def startup_event():
    print("🔁 Loading precomputed dataset...")
    load_and_index_dataset()
    print("✅ Dataset ready.")
