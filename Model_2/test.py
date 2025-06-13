import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cpu")

model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint).to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

text_list = ["This is a test sentence for embedding."]
encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

with torch.no_grad():
    model_output = model(**encoded_input)
    
embedding = cls_pooling(model_output)
print("Embedding shape:", embedding.shape)
