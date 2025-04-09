import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

# Removed import of ModelArgs; using TransformerConfig below instead
# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )

class TransformerConfig:
    """Configuration for the Transformer model."""
    def __init__(
        self,
        vocab_size: int = 200000,
        block_size: int = 2048,   # maximum sequence length
        n_layer: int = 32,        # number of transformer blocks
        n_head: int = 32,         # number of attention heads
        n_embd: int = 4096,       # embedding dimension
        dropout: float = 0.0,     # dropout rate
        norm_eps: float = 1e-6,   # epsilon for RMSNorm
        multiple_of: int = 1,     # multiple for adjusting hidden dim
        ffn_dim_multiplier: Optional[float] = None,  # optional FFN dim multiplier
        n_kv_head: Optional[int] = None,  # number of key/value heads (if different from n_head)
        rope_theta: float = 10000.0,      # base frequency for rotary embeddings
        use_scaled_rope: bool = False,    # whether to apply scaled rotary embeddings
        max_batch_size: int = 1,         # maximum batch size for caching
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.n_kv_head = n_kv_head
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope
        self.max_batch_size = max_batch_size

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize then scale
        return self._norm(x.float()).type_as(x) * self.weight

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Scaling factors for extended rotary embeddings (if use_scaled_rope=True)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original context length baseline
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    # Convert to complex cosine/sine form for rotary embedding application
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # shape: (end, dim/2)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Adjust shape of freqs_cis for broadcasting to tensor x
    ndim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # Create shape [1, seq_len, 1, ..., head_dim] to multiply with x
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Apply rotary embedding to query and key
    # Convert to complex
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Broadcast complex exponentials to match q,k shapes and multiply
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    # Return to original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensor along head dimension n_rep times."""
    bs, slen, n_kv_head, head_dim = x.shape
    if n_rep == 1:
        return x  # no repetition needed
    # Expand and reshape to repeat the key/value heads
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_head, n_rep, head_dim)
        .reshape(bs, slen, n_kv_head * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Determine number of key/value heads (for grouped QKV, default equal to n_head)
        self.n_kv_head = config.n_head if config.n_kv_head is None else config.n_kv_head
        # Support for model parallel (world_size); assume 1 if not using model parallel
        try:
            import fairscale.nn.model_parallel.initialize as fs_init
            world_size = fs_init.get_model_parallel_world_size()
        except ImportError:
            world_size = 1
        # Local heads per partition (for model parallel or single device)
        self.n_local_head = config.n_head // world_size
        self.n_local_kv_head = self.n_kv_head // world_size
        # Factor to repeat keys/values if n_kv_head < n_head
        self.n_rep = self.n_local_head // self.n_local_kv_head
        # Head dimension
        self.head_dim = config.n_embd // config.n_head

        # Query, Key, Value projection layers
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Output projection layer
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)

        # Allocate key and value cache tensors (on CPU by default, will move to device as needed)
        self.cache_k = torch.zeros(config.max_batch_size, config.block_size, self.n_local_kv_head, self.head_dim)
        self.cache_v = torch.zeros(config.max_batch_size, config.block_size, self.n_local_kv_head, self.head_dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        B, T, C = x.size()
        # Compute query, key, value projections
        q = self.q_proj(x)  # shape (B, T, n_head * head_dim)
        k = self.k_proj(x)  # shape (B, T, n_kv_head * head_dim)
        v = self.v_proj(x)  # shape (B, T, n_kv_head * head_dim)
        # Reshape into [batch, seq, heads, head_dim] for rotary embedding and attention
        q = q.view(B, T, self.n_local_head, self.head_dim)
        k = k.view(B, T, self.n_local_kv_head, self.head_dim)
        v = v.view(B, T, self.n_local_kv_head, self.head_dim)
        # Apply rotary position embeddings to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        # Move caches to the same device as q (happens automatically after first use)
        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(q)
        # Store new keys and values in cache at positions [start_pos : start_pos+T]
        self.cache_k[:B, start_pos : start_pos + T] = k
        self.cache_v[:B, start_pos : start_pos + T] = v
        # Retrieve all cached keys and values up to the current position
        keys = self.cache_k[:B, : start_pos + T]    # shape (B, start_pos+T, n_local_kv_head, head_dim)
        values = self.cache_v[:B, : start_pos + T]  # shape (B, start_pos+T, n_local_kv_head, head_dim)
        # If using fewer KV heads than query heads, repeat keys and values to match head count
        if self.n_rep > 1:
            keys = repeat_kv(keys, self.n_rep)     # shape (B, start_pos+T, n_local_head, head_dim)
            values = repeat_kv(values, self.n_rep)  # shape (B, start_pos+T, n_local_head, head_dim)
        # Transpose to shape (B, head, seq_len, head_dim) for attention computation
        q = q.transpose(1, 2)         # (B, n_local_head, T, head_dim)
        keys = keys.transpose(1, 2)   # (B, n_local_head, start_pos+T, head_dim)
        values = values.transpose(1, 2)  # (B, n_local_head, start_pos+T, head_dim)
        # Scaled dot-product self-attention
        scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # (B, n_local_head, T, start_pos+T)
        if mask is not None:
            scores = scores + mask  # apply causal mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)  # normalize attention weights
        # Multiply attention weights by values
        attn_output = torch.matmul(scores, values)  # (B, n_local_head, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_local_head*head_dim)
        # Final linear projection back to embedding dimension
        return self.c_proj(attn_output)
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Determine hidden dimension for the FFN (with LLaMA-style adjustment)
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)  # 2/3 factor
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        # Feed-forward network layers (SwiGLU style: two projection layers in, one out)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU activation on w1, elementwise product with w3, then linear projection w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """Single Transformer decoder block with pre-LayerNorm, self-attention, and MLP."""
    def __init__(self, layer_id: int, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attention = Attention(config)
        self.mlp = FeedForward(config)
        self.layer_id = layer_id
        # Layer normalization (RMSNorm) layers for attention and FFN
        self.ln1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ln2 = RMSNorm(config.n_embd, eps=config.norm_eps)
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # LayerNorm at the start of the block (pre-normalization)
        x_norm = self.ln1(x)
        # Self-attention layer
        y = self.attention(x_norm, start_pos, freqs_cis, mask)
        # Residual connection
        x = x + y
        # MLP block
        x_norm2 = self.ln2(x)
        y2 = self.mlp(x_norm2)
        # Residual connection
        x = x + y2
        return x

class TransformerModel(nn.Module):
    """Transformer Language Model with rotary embeddings (LLaMA-style) and caching."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Token embedding layer
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Dropout on embeddings
        self.drop = nn.Dropout(config.dropout)
        # List of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(layer_id, config) for layer_id in range(config.n_layer)])
        # Final layer normalization
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        # Output head (logits for vocab)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # (Optional) tie embedding and output weights for efficiency
        # self.head.weight = self.tok_emb.weight  # uncomment if weight tying is desired
        # Precompute rotary embedding frequencies (complex values)
        self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size * 2, config.rope_theta, config.use_scaled_rope)
    def forward(self, idx: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        :param idx: Tensor of token IDs with shape (B, T).
        :param start_pos: Starting position (0 for sequence start, >0 for continuation).
        :return: Logits tensor of shape (B, T, vocab_size).
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block_size"
        # Token embeddings + dropout
        x = self.tok_emb(idx)            # shape (B, T, n_embd)
        x = self.drop(x)
        # Select precomputed rotary freqs for current positions
        self.freqs_cis = self.freqs_cis.to(x.device)  # move to device
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        # Causal mask for attention (allow full access to previous tokens)
        mask = None
        if T > 1:
            mask = torch.full((T, T), float("-inf"), device=idx.device)
            mask = torch.triu(mask, diagonal=1)
            # Fix for potential NaNs on certain devices (e.g., MPS)
            if mask.device.type == "mps":
                mask = torch.nan_to_num(mask, nan=0.0)
            # Prepend zeros for past tokens (so new tokens can attend to all prev tokens)
            mask = torch.hstack([torch.zeros((T, start_pos), device=idx.device), mask]).type_as(x)
        # Apply each Transformer block
        for block in self.blocks:
            x = block(x, start_pos, freqs_cis, mask)
        # Final layer norm
        x = self.ln_f(x)
        # Compute logits
        logits = self.head(x).float()  # shape (B, T, vocab_size)
        return logits
