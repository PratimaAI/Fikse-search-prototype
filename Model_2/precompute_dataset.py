import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import spacy

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

def main():
    device = torch.device("cpu")
    model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint).to(device)

    df = pd.read_csv("Dataset_categories.csv")
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
    
    # Save the dataset with embeddings
    dataset.save_to_disk("precomputed_dataset")
    
    print("✅ Dataset and FAISS index saved successfully!")

if __name__ == "__main__":
    main()