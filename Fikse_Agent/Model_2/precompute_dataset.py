import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
import math

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def lemmatize_and_lower(text):
    if not isinstance(text, str):
        return text
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def preprocess_batch(batch):
    # Apply lemmatization and lowercasing to all string fields
    processed = {}
    for key, values in batch.items():
        processed[key] = [
            lemmatize_and_lower(v) if isinstance(v, str) else v
            for v in values
        ]
    return processed

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

def main():
    device = torch.device("cpu")
    model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint).to(device)

    # Load and clean the dataset
    df = pd.read_csv("Dataset_agent_new.csv", delimiter=";")
    
    # Clean missing values and ensure proper data types
    df = df.fillna("")  # Replace NaN with empty string for text columns
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)  # Convert to numeric, fill NaN with 0
    df["Estimated time in hours"] = pd.to_numeric(df["Estimated time in hours"], errors="coerce").fillna(0)  # Convert to numeric, fill NaN with 0
    
    # Ensure all text columns are strings
    text_columns = ["Type of Repairer", "Type of category", "Type of garment in category", "Service", "Description"]
    for col in text_columns:
        df[col] = df[col].astype(str)
    
    dataset = Dataset.from_pandas(df)

    # Step 1: Preprocess text columns
    dataset = dataset.map(preprocess_batch, batched=True)

    # Step 2: Create "text" column by concatenation
    dataset = dataset.map(concatenate_text, batched=True)

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

if __name__ == "__main__":
    main()