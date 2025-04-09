import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2VecNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_emb = nn.Embedding(vocab_size, embed_dim)
        self.context_emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, pos_context, neg_contexts):
        center_emb = self.center_emb(center)              # (B, D)
        pos_emb = self.context_emb(pos_context)           # (B, D)
        neg_emb = self.context_emb(neg_contexts)          # (B, K, D)

        # Positive loss
        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative loss
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        return -(pos_loss + neg_loss).mean()