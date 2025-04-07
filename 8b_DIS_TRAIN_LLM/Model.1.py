import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    """Configuration for the Transformer model."""
    def __init__(self, vocab_size=200000, block_size=2048, n_layer=32, n_head=32, n_embd=4096, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size  # maximum sequence length
        self.n_layer = n_layer       # number of transformer blocks
        self.n_head = n_head         # number of attention heads
        self.n_embd = n_embd         # embedding dimensionality
        self.dropout = dropout       # dropout rate

class TransformerBlock(nn.Module):
    """Single Transformer decoder block with pre-LayerNorm, self-attention, and MLP."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Self-attention: combine queries, keys, values in one linear for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Feed-forward network
        self.mlp_fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        # LayerNorms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Causal mask to ensure attention only sees past and current positions
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        # LayerNorm at the start (pre-LN Transformer)
        x_norm = self.ln1(x)
        # Compute query, key, value all at once
        qkv = self.c_attn(x_norm)  # shape (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v each have shape (B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Self-attention: Q*K^T scaled by sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply causal mask: prevent attention to future positions
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # Attention output
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Output projection + dropout
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        # Residual connection
        x = x + y

        # MLP block
        x_norm2 = self.ln2(x)
        y2 = self.mlp_fc1(x_norm2)
        y2 = F.gelu(y2)
        y2 = self.mlp_fc2(y2)
        y2 = self.resid_dropout(y2)
        # Residual connection
        x = x + y2
        return x

class TransformerModel(nn.Module):
    """8B-parameter Transformer Language Model."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Token embeddings and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # Final normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Output head (language model logits)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tie token embedding and output weights for efficiency
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        """
        Forward pass for the model.
        :param idx: Tensor of token IDs with shape (B, T).
        :return: Logits of shape (B, T, vocab_size).
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block_size"
        # Token and positional embeddings
        token_embeddings = self.tok_emb(idx)        # (B, T, n_embd)
        position_ids = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        position_embeddings = self.pos_emb(position_ids)  # (1, T, n_embd)
        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        x = self.drop(x)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # Final layer norm
        x = self.ln_f(x)
        # Output logits
        logits = self.head(x)  # (B, T, vocab_size)
        return logits