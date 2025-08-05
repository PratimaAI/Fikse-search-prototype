# This is the search engine (working individually). This will be working on specific phrase or words.

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

def load_shared_config():
    """Load dataset configuration from JSON file"""
    try:
        with open("dataset.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  dataset.json not found. Using default configuration.")
        return {
            "search_priorities": {
                "exact_service_name": "service_name",
                "partial_service_name": "service_name", 
                "service_type": "service_type",
                "description": "description",
                "item_name": "item_name",
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

    corrected_query = correct_query(q)
    normalized_query = lemmatize_and_lower(corrected_query)
    target_price = extract_price(corrected_query)
    
    # STAGE 1: Get semantic candidates
    print(f"🔍 Searching for: '{q}' (normalized: '{normalized_query}')")
    query_embedding = embed_text([normalized_query])[0]
    scores, samples = dataset.get_nearest_examples(
        index_name="embeddings", query=query_embedding, k=100
    )
    
    all_results = pd.DataFrame(samples)
    all_results["similarity_score"] = scores
    
    # STAGE 2: Enhanced keyword matching with damage detection
    search_terms = [term.lower() for term in q.split()]
    
    # Detect damage-related terms
    damage_terms = ["damaged", "broken", "torn", "ripped", "hole", "stain", "repair", "fix", "mend"]
    has_damage_context = any(term in search_terms for term in damage_terms)
    
    # Detect fabric-specific terms
    fabric_terms = ["silk", "cotton", "wool", "leather", "denim", "linen", "polyester"]
    has_fabric_context = any(term in search_terms for term in fabric_terms)
    
    print(f"🎯 Looking for terms: {search_terms}")
    if has_damage_context:
        print(f"🔧 Damage context detected: {[term for term in search_terms if term in damage_terms]}")
    if has_fabric_context:
        print(f"🧵 Fabric context detected: {[term for term in search_terms if term in fabric_terms]}")
    
    exact_service_matches = []
    partial_service_matches = []
    service_type_matches = []
    description_matches = []
    item_name_matches = []
    category_matches = []
    business_type_matches = []
    semantic_only = []
    
    # Get column names from shared configuration
    priorities = config["search_priorities"]
    priority_order = config.get("search_priority_order", [
        "exact_service_name", "partial_service_name", "service_type", 
        "description", "item_name", "category", "business_type"
    ])
    
    # Categorize matches by relevance
    for i, row in all_results.iterrows():
        # Get field values using shared configuration
        service_name_lower = str(row.get(priorities["exact_service_name"], "")).lower()
        service_type_lower = str(row.get(priorities["service_type"], "")).lower()
        description_lower = str(row.get(priorities["description"], "")).lower()
        item_name_lower = str(row.get(priorities["item_name"], "")).lower()
        business_type_lower = str(row.get(priorities["business_type"], "")).lower()
        category_lower = str(row.get(priorities.get("category", "category"), "")).lower()
        
        match_found = False
        match_details = []
        
        # Check each search term
        for term in search_terms:
            # Highest priority: Exact service name match
            if term == service_name_lower:
                exact_service_matches.append((row, scores[i], f"exact_service_name:{term}"))
                match_found = True
                break
            # High priority: Partial service name match
            elif term in service_name_lower:
                partial_service_matches.append((row, scores[i], f"partial_service_name:{term}"))
                match_found = True
                break
            # High priority: Service type match (REPAIR, ALTERATION, etc.)
            elif term in service_type_lower:
                # Enhanced service type matching with context awareness
                if has_damage_context and "repair" in service_type_lower:
                    # Boost repair services when damage is mentioned
                    service_type_matches.append((row, scores[i] + 5, f"service_type:repair:{term}"))
                elif has_damage_context and "alteration" in service_type_lower:
                    # Lower priority for alterations when damage is mentioned
                    service_type_matches.append((row, scores[i] - 3, f"service_type:alteration:{term}"))
                else:
                    service_type_matches.append((row, scores[i], f"service_type:{term}"))
                match_found = True
                match_details.append(f"service_type:{term}")
            # Medium priority: Description match
            elif term in description_lower:
                description_matches.append((row, scores[i], f"description:{term}"))
                match_found = True
                match_details.append(f"description:{term}")
            # Lower priority: Item name match
            elif term in item_name_lower:
                item_name_matches.append((row, scores[i], f"item_name:{term}"))
                match_found = True
                match_details.append(f"item_name:{term}")
            # Lower priority: Category match
            elif term in category_lower:
                category_matches.append((row, scores[i], f"category:{term}"))
                match_found = True
                match_details.append(f"category:{term}")
            # Lowest priority: Business type match
            elif term in business_type_lower:
                business_type_matches.append((row, scores[i], f"business_type:{term}"))
                match_found = True
                match_details.append(f"business_type:{term}")
        
        if not match_found:
            semantic_only.append((row, scores[i], "semantic_only"))
    
    print(f"📊 Match breakdown:")
    print(f"  - Exact service name matches: {len(exact_service_matches)}")
    print(f"  - Partial service name matches: {len(partial_service_matches)}")
    print(f"  - Service type matches: {len(service_type_matches)}")
    print(f"  - Description matches: {len(description_matches)}")
    print(f"  - Item name matches: {len(item_name_matches)}")
    print(f"  - Category matches: {len(category_matches)}")
    print(f"  - Business type matches: {len(business_type_matches)}")
    print(f"  - Semantic only: {len(semantic_only)}")
    
    # STAGE 3: Combine results with smart prioritization
    final_results = []
    
    # Define match type priorities (higher number = higher priority)
    match_type_priorities = {
        "exact_service_name": 100,
        "partial_service_name": 90,
        "service_type": 80,
        "description": 70,
        "item_name": 60,
        "category": 50,
        "business_type": 40,
        "semantic": 10
    }
    
    # Boost priorities based on context
    if has_damage_context:
        # Boost repair-related services when damage is mentioned
        repair_boost = 20
        match_type_priorities["service_type"] += repair_boost
        print(f"🔧 Boosting repair services by {repair_boost} points")
    
    if has_fabric_context:
        # Boost description matches when fabric is mentioned
        fabric_boost = 15
        match_type_priorities["description"] += fabric_boost
        print(f"🧵 Boosting fabric-specific matches by {fabric_boost} points")
    
    # Collect all matches with their priorities
    all_matches = []
    
    # Add all match groups with their priorities
    match_groups = [
        (exact_service_matches, "exact_service_name"),
        (partial_service_matches, "partial_service_name"), 
        (service_type_matches, "service_type"),
        (description_matches, "description"),
        (item_name_matches, "item_name"),
        (category_matches, "category"),
        (business_type_matches, "business_type"),
        (semantic_only, "semantic")
    ]
    
    for match_group, match_type in match_groups:
        for row, score, match_detail in match_group:
            # Calculate combined score: match_priority + normalized_similarity_score
            match_priority = match_type_priorities.get(match_type, 0)
            # Normalize similarity score to 0-1 range (assuming scores are typically 20-35)
            normalized_score = max(0, min(1, (score - 20) / 15))  # Normalize 20-35 to 0-1
            combined_score = match_priority + normalized_score
            
            all_matches.append((row, score, match_detail, match_type, combined_score))
    
    # Sort by combined score (highest first)
    all_matches.sort(key=lambda x: x[4], reverse=True)
    
    # Take top 10 results
    for row, score, match_detail, match_type, combined_score in all_matches[:10]:
        row_dict = row.to_dict()
        row_dict["similarity_score"] = float(score)
        row_dict["match_type"] = match_type
        row_dict["match_detail"] = match_detail
        row_dict["search_terms"] = search_terms
        final_results.append(row_dict)
    
    # Apply price filter if needed
    if target_price:
        price_col = config["price_column"]
        def is_price_match(result):
            try:
                return abs(float(result.get(price_col, 0)) - target_price) <= 50
            except:
                return False
        final_results = [r for r in final_results if is_price_match(r)]
        print(f"💰 Price filter applied: {target_price} ± 50")
    
    print(f"🎯 Returning {len(final_results)} results")
    for i, result in enumerate(final_results[:5]):
        service_name = result.get(priorities["exact_service_name"], "Unknown Service")
        print(f"  {i+1}. {service_name} ({result['match_type']}) - Score: {result['similarity_score']:.2f}")
    
    return final_results[:10]

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
