import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import math
import json
import os

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def lemmatize_and_lower(text):
    if not isinstance(text, str):
        return text
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def preprocess_batch(batch):
    # Apply lemmatize_and_lower to all string fields
    processed = {}
    for key, values in batch.items():
        processed[key] = [
            lemmatize_and_lower(v) if isinstance(v, str) else v
            for v in values
        ]
    return processed

def concatenate_text(batch, text_columns):
    texts = []
    for i in range(len(batch[text_columns[0]])):
        text_parts = []
        for col in text_columns:
            text_parts.append(str(batch[col][i]))
        text = "\n".join(text_parts)
        texts.append(lemmatize_and_lower(text))
    return {"text": texts}

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list, tokenizer, model, device):
    encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    return cls_pooling(model_output)

def embed_batch(batch, tokenizer, model, device):
    embeddings_tensor = get_embeddings(batch["text"], tokenizer, model, device)
    embeddings_np = embeddings_tensor.detach().cpu().numpy()
    return {"embeddings": embeddings_np}

# Added: Clean invalid float values from embeddings (NaN, inf)
def sanitize_floats(obj):
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        else:
            return obj
    else:
        return obj

def detect_column_mapping(df):
    """
    Automatically detect column mapping based on common patterns
    Returns a dictionary mapping standard names to actual column names
    """
    columns = df.columns.tolist()
    mapping = {}
    
    # Common patterns for different column types
    patterns = {
        'business_type': ['business_type', 'business', 'repairer', 'repairer_type', 'type_of_repairer', 'provider', 'vendor'],
        'category': ['category', 'item_category', 'item_category_name', 'type_of_category', 'category_name', 'product_category'],
        'item_name': ['item_name', 'item', 'garment', 'garment_type', 'type_of_garment', 'product_name', 'product'],
        'service_name': ['service_name', 'service'],
        'service_type': ['service_type', 'service_category'],
        'description': ['description', 'service_description', 'service_details'],
        'price': ['price', 'service_price', 'cost', 'amount', 'estimated_price', 'rate', 'fee'],
        'hours': ['hours', 'hours_estimate', 'estimated_hours', 'time', 'estimated_time', 'duration', 'work_hours']
    }
    
    for standard_name, possible_names in patterns.items():
        for col in columns:
            if col.lower() in [name.lower() for name in possible_names]:
                mapping[standard_name] = col
                break
    
    # If no mapping found for required fields, use first available columns
    required_fields = ['business_type', 'category', 'item_name', 'service_name', 'description', 'price']
    for field in required_fields:
        if field not in mapping and columns:
            mapping[field] = columns[0]  # Fallback to first column
    
    print(f"📊 Detected column mapping: {mapping}")
    return mapping

def create_standardized_dataset(df, column_mapping):
    """
    Create a standardized dataset with consistent column names
    This ensures all downstream modules work with the same interface
    """
    standardized_df = df.copy()
    
    # Create standardized columns
    standardized_columns = {
        'business_type': column_mapping.get('business_type', 'business_type'),
        'category': column_mapping.get('category', 'category'),
        'item_name': column_mapping.get('item_name', 'item_name'),
        'service_name': column_mapping.get('service_name', 'service_name'),
        'service_type': column_mapping.get('service_type', 'service_type'),
        'description': column_mapping.get('description', 'description'),
        'price': column_mapping.get('price', 'price'),
        'hours': column_mapping.get('hours', 'hours')
    }
    
    # Rename columns to standard names
    rename_mapping = {v: k for k, v in standardized_columns.items() if v in df.columns}
    standardized_df = standardized_df.rename(columns=rename_mapping)
    
    # Add missing columns with defaults
    for col in ['business_type', 'category', 'item_name', 'service_name', 'service_type', 'description', 'price', 'hours']:
        if col not in standardized_df.columns:
            if col in ['price', 'hours']:
                standardized_df[col] = 0
            else:
                standardized_df[col] = "Unknown"
    
    return standardized_df

def save_shared_config(column_mapping, dataset_info):
    """
    Save comprehensive shared configuration for all modules
    """
    config = {
        "dataset_info": dataset_info,
        "standard_columns": [
            "business_type", "category", "item_name", "service_name", 
            "service_type", "description", "price", "hours"
        ],
        "search_priorities": {
            "exact_service_name": "service_name",
            "partial_service_name": "service_name", 
            "service_type": "service_type",
            "description": "description",
            "item_name": "item_name",
            "business_type": "business_type",
            "category": "category"
        },
        "price_column": "price",
        "hours_column": "hours",
        "text_columns": ["business_type", "category", "item_name", "service_name", "service_type", "description"],
        "search_priority_order": [
            "exact_service_name",
            "partial_service_name", 
            "service_type",
            "description",
            "item_name",
            "category",
            "business_type"
        ]
    }
    
    with open("dataset.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Dataset configuration saved to dataset.json")

def main():
    device = torch.device("cpu")
    model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint).to(device)

    # Load and clean the dataset
    df = pd.read_csv("HUB_Service.csv", delimiter=";")
    
    # Auto-detect column mapping
    column_mapping = detect_column_mapping(df)
    
    # Create standardized dataset
    standardized_df = create_standardized_dataset(df, column_mapping)
    
    # Clean missing values and ensure proper data types
    standardized_df = standardized_df.fillna("")  # Replace NaN with empty string for text columns
    
    # Convert price and hours to numeric
    standardized_df["price"] = pd.to_numeric(standardized_df["price"], errors="coerce").fillna(0)
    standardized_df["hours"] = pd.to_numeric(standardized_df["hours"], errors="coerce").fillna(0)
    
    # Ensure all text columns are strings
    text_columns = ["business_type", "category", "item_name", "service_name", "service_type", "description"]
    for col in text_columns:
        standardized_df[col] = standardized_df[col].astype(str)
    
    # Save shared configuration
    dataset_info = {
        "original_columns": list(df.columns),
        "standardized_columns": list(standardized_df.columns),
        "total_rows": len(standardized_df),
        "file_name": "HUB_Service.csv",
        "column_mapping": column_mapping
    }
    save_shared_config(column_mapping, dataset_info)
    
    dataset = Dataset.from_pandas(standardized_df)

    # Step 1: Preprocess text columns
    dataset = dataset.map(preprocess_batch, batched=True)

    # Step 2: Create "text" column by concatenation
    dataset = dataset.map(lambda batch: concatenate_text(batch, text_columns), batched=True)

    # Step 3: Embed with transformer
    dataset = dataset.map(lambda batch: embed_batch(batch, tokenizer, model, device), batched=True, batch_size=1)

    # Step 4: Save FAISS index and dataset
    dataset.add_faiss_index(column="embeddings")
    
    # Save the FAISS index separately
    dataset.get_index("embeddings").save("faiss.index")
    
    # Drop the FAISS index before saving (modifies dataset in place)
    dataset.drop_index("embeddings")
    
    # Sanitize embeddings before saving to disk
    dataset = dataset.map(lambda x: {"embeddings": sanitize_floats(x["embeddings"])}, batched=True)

    # Save the dataset with embeddings
    dataset.save_to_disk("precomputed_dataset")
    
    print("✅ Dataset and FAISS index saved successfully!")
    print(f"📋 Dataset configuration saved to dataset.json")
    print(f"🔧 Standardized columns: {list(standardized_df.columns)}")

if __name__ == "__main__":
    main()