import torch
import numpy as np
from model import Word2VecNegSampling
import tiktoken 
from sklearn.metrics.pairwise import cosine_similarity

with open("preprocessed/meta.txt") as f:
    vocab_size = int(f.readline())

enc = tiktoken.get_encoding("o200k_base") 

model = Word2VecNegSampling(vocab_size, 600)
model.load_state_dict(torch.load("embedding_model.pth"))
model.eval()

embedding_weights = model.center_emb.weight.data.cpu().numpy()
query_token = enc.encode("king")[0] 
query = embedding_weights[query_token]

sims = cosine_similarity([query], embedding_weights)[0]
topk = sims.argsort()[-10:][::-1]

print("Top 10 similar tokens:")
for i in topk:
    print(enc.decode([i]), f"({sims[i]:.3f})")
