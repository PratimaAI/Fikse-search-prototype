# This is the search engine for agent (this will be working with whole sentences)

import sys
print("Python executable:", sys.executable)

import os
import re
import json
import pandas as pd
import numpy as np
import torch
from datasets import load_from_disk
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModel
from symspellpy.symspellpy import SymSpell, Verbosity
from rapidfuzz import fuzz

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

# === New Search Functions ===
def detect_service_type_from_query(query):
    """Detect service type from user query keywords"""
    query_lower = query.lower()
    
    # REPAIR detection (65% of dataset)
    repair_keywords = ["damaged", "broken", "torn", "hole", "holes", "repair", "fix", "mend", "patching"]
    if any(keyword in query_lower for keyword in repair_keywords):
        return "repair"
    
    # ALTERATION detection (26% of dataset)
    alteration_keywords = ["shorten", "lengthen", "take in", "take out", "alteration", "alter", "adjust"]
    if any(keyword in query_lower for keyword in alteration_keywords):
        return "alteration"
    
    # DRY CLEANING detection (8% of dataset)
    dry_clean_keywords = ["dryclean", "dry cleaning", "clean", "washing", "polishing"]
    if any(keyword in query_lower for keyword in dry_clean_keywords):
        return "dry cleaning"
    
    return None

def fuzzy_match(query_term, target_text, threshold=85):
    """Perform fuzzy matching with specified threshold"""
    if not target_text or pd.isna(target_text):
        return 0
    return fuzz.partial_ratio(query_term.lower(), str(target_text).lower())

def calculate_combined_score(semantic_score, fuzzy_score, match_type, match_priority):
    """Calculate combined score from semantic, fuzzy, and match type"""
    # Normalize semantic score (typically 20-35 range)
    normalized_semantic = max(0, min(1, (semantic_score - 20) / 15))
    
    # Normalize fuzzy score (0-100)
    normalized_fuzzy = fuzzy_score / 100
    
    # Combined score formula
    combined_score = (
        normalized_semantic * 0.3 +      # 30% semantic
        normalized_fuzzy * 0.4 +         # 40% fuzzy matching
        match_priority * 0.3             # 30% match type priority
    )
    
    return combined_score

def extract_search_terms(query):
    """Extract meaningful search terms from query"""
    # Remove common words that don't help with search
    stop_words = {"help", "me", "create", "an", "order", "for", "this", "that", "the", "a", "my", "want", "to", "i"}
    
    # Clean and split query
    cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
    terms = [term for term in cleaned_query.split() if term not in stop_words and len(term) > 2]
    
    return terms

def load_shared_config():
    """Load dataset configuration from JSON file"""
    try:
        with open("dataset.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  dataset.json not found. Using default configuration.")
        return {
            "search_priorities": {
                "service_type": "service_type",
                "item_name": "item_name",
                "category": "category",
                "service_name": "service_name",
                "description": "description",
                "business_type": "business_type"
            },
            "price_column": "price",
            "hours_column": "hours",
            "standard_columns": [
                "business_type", "category", "item_name", "service_name", 
                "service_type", "description", "price", "hours"
            ]
        }

# === Define global variables ===
dataset = None
config = None

def load_and_index_dataset():
    global dataset, config
    print("Loading precomputed dataset from disk...")
    dataset = load_from_disk("precomputed_dataset")
    dataset.load_faiss_index("embeddings", "faiss.index")
    config = load_shared_config()
    print("✅ Dataset loaded.")
    print(f"📋 Using shared configuration")

# === Routes ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/search")
def search_api(q: str):
    if dataset is None:
        return {"error": "Dataset not loaded. Please restart server."}

    print(f"🔍 NEW SEARCH REQUEST: '{q}'")
    
    # Step 1: Query preprocessing
    corrected_query = correct_query(q.lower())
    search_terms = extract_search_terms(corrected_query)
    detected_service_type = detect_service_type_from_query(corrected_query)
    
    print(f"📝 Query processing:")
    print(f"  - Original: '{q}'")
    print(f"  - Corrected: '{corrected_query}'")
    print(f"  - Search terms: {search_terms}")
    print(f"  - Detected service type: {detected_service_type}")
    
    # Step 2: Semantic search on FULL dataset first (no filtering yet)
    print(f"🔍 Stage 1: Semantic search on full dataset...")
    print(f"  → Getting query embeddings...")
    query_embedding = embed_text([corrected_query])[0]
    scores, samples = dataset.get_nearest_examples(
        index_name="embeddings", query=query_embedding, k=100
    )
    
    all_results = pd.DataFrame(samples)
    all_results["similarity_score"] = scores
    
    print(f"  - Found {len(all_results)} semantic candidates")
    
    # Step 3: Filter by service type AFTER semantic search
    if detected_service_type:
        print(f"🎯 Stage 2: Filtering semantic results by service type '{detected_service_type}'")
        before_filter = len(all_results)
        all_results = all_results[all_results['service_type'] == detected_service_type.lower()]
        after_filter = len(all_results)
        print(f"  - Filtered from {before_filter} to {after_filter} {detected_service_type} candidates")
        
        # If no results after filtering, fall back to all semantic results
        if len(all_results) == 0:
            print(f"  - No {detected_service_type} results found, using all semantic candidates")
            scores, samples = dataset.get_nearest_examples(
                index_name="embeddings", query=query_embedding, k=100
            )
            all_results = pd.DataFrame(samples)
            all_results["similarity_score"] = scores
    else:
        print(f"🎯 Stage 2: No service type detected, using all semantic candidates")
    
    # Step 4: Fuzzy and exact matching with combined scoring
    print(f"🎯 Stage 3: Fuzzy/exact matching with combined scoring...")
    
    # Define match priorities
    match_priorities = {
        "exact_service_type": 100,
        "exact_item_name": 90,
        "exact_category": 85,
        "exact_service_name": 80,
        "exact_description": 75,
        "fuzzy_service_type": 70,
        "fuzzy_item_name": 65,
        "fuzzy_category": 60,
        "fuzzy_service_name": 55,
        "fuzzy_description": 50,
        "semantic_only": 10
    }
    
    scored_results = []
    for idx, row in all_results.iterrows():
        best_match_type = "semantic_only"
        best_fuzzy_score = 0
        best_match_priority = match_priorities["semantic_only"]
        
        for term in search_terms:
            service_type = str(row.get('service_type', '')).lower()
            service_name = str(row.get('service_name', '')).lower()
            item_name = str(row.get('item_name', '')).lower()
            item_category = str(row.get('category', '')).lower()
            description = str(row.get('description', '')).lower()
            
            # Exact matches
            if term == service_type:
                best_match_type = "exact_service_type"
                best_fuzzy_score = 100
                best_match_priority = match_priorities["exact_service_type"]
                break
            elif term == service_name:
                best_match_type = "exact_service_name"
                best_fuzzy_score = 100
                best_match_priority = match_priorities["exact_service_name"]
                break
            elif term == item_name:
                if best_match_priority < match_priorities["exact_item_name"]:
                    best_match_type = "exact_item_name"
                    best_fuzzy_score = 100
                    best_match_priority = match_priorities["exact_item_name"]
            elif term == item_category:
                if best_match_priority < match_priorities["exact_category"]:
                    best_match_type = "exact_category"
                    best_fuzzy_score = 100
                    best_match_priority = match_priorities["exact_category"]
            else: # Fuzzy matches
                fuzzy_scores = {
                    "fuzzy_service_type": fuzzy_match(term, service_type),
                    "fuzzy_service_name": fuzzy_match(term, service_name),
                    "fuzzy_item_name": fuzzy_match(term, item_name),
                    "fuzzy_category": fuzzy_match(term, item_category),
                    "fuzzy_description": fuzzy_match(term, description)
                }
                for match_type, score in fuzzy_scores.items():
                    if score >= 85 and match_priorities[match_type] > best_match_priority:
                        best_match_type = match_type
                        best_fuzzy_score = score
                        best_match_priority = match_priorities[match_type]
        
        # Calculate combined score
        combined_score = calculate_combined_score(
            semantic_score=row['similarity_score'],
            fuzzy_score=best_fuzzy_score,
            match_type=best_match_type,
            match_priority=best_match_priority
        )
        
        result_dict = row.to_dict()
        result_dict.update({
            "combined_score": combined_score,
            "match_type": best_match_type,
            "fuzzy_score": best_fuzzy_score,
            "search_terms": search_terms,
            "detected_service_type": detected_service_type
        })
        scored_results.append(result_dict)
    
    # Step 5: Sort by combined score and return top 10
    scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
    final_results = scored_results[:10]
    
    print(f"🎯 Stage 4: Final results ({len(final_results)} items)")
    for i, result in enumerate(final_results[:5]):
        service_name = result.get('service_name', 'Unknown')
        match_type = result.get('match_type', 'unknown')
        combined_score = result.get('combined_score', 0)
        print(f"  {i+1}. {service_name} ({match_type}) - Score: {combined_score:.2f}")
    
    return final_results

@app.get("/dataset-config")
def get_dataset_config():
    """Serve dataset configuration to frontend"""
    if config is None:
        return {"error": "Dataset configuration not loaded"}
    return config

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
    
    priorities = config["search_priorities"]
    service_name_col = priorities["exact_service_name"]
    description_col = priorities["description"]
    
    for i in range(min(5, len(results))):
        row = results.iloc[i]
        debug_entry = {
            "service_name": row.get(service_name_col, "Unknown"),
            "service_name_lower": str(row.get(service_name_col, "")).lower(),
            "service_name_lemmatized": lemmatize_and_lower(str(row.get(service_name_col, ""))),
            "description": row.get(description_col, ""),
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
        "shared_config": config,
        "sample_entries": debug_info
    }

@app.on_event("startup")
def startup_event():
    print("🔁 Loading precomputed dataset...")
    load_and_index_dataset()
    print("✅ Dataset ready.")
