import pandas as pd
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("Dataset_Categories.csv")
df = df.fillna("")

def format_record(row):
    return f"{row['Type of Repairer']} {row['Type of category']} {row['Type of garment in category']} {row['Service']} {row['Description']} {row['Price']}"

texts = df.apply(format_record, axis=1).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

with open("fikse_data.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype(np.float32))

faiss.write_index(index, "fikse_index.faiss")

print("✅ Finished: Data saved as 'fikse_data.pkl' and 'fikse_index.faiss'")
