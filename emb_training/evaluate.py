import torch
import numpy as np
from model import Word2VecNegSampling

with open("preprocessed/meta.txt") as f:
    vocab_size = int(f.readline())

model = Word2VecNegSampling(vocab_size, 300)
model.load_state_dict(torch.load("embedding_model.pth"))
model.eval()

# Cosine similarity search
embedding_weights = model.center_emb.weight.data.cpu().numpy()
query = embedding_weights[ enc.encode("king")[0] ]

from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity([query], embedding_weights)[0]
topk = sims.argsort()[-10:][::-1]

print("Top 10 similar tokens:")
for i in topk:
    print(enc.decode([i]), f"({sims[i]:.3f})")
