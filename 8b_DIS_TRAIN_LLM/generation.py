import os
import torch
import torch.nn.functional as F
import deepspeed
import torch.distributed as dist
from safetensors.torch import load_file
from model import TransformerConfig, TransformerModel
from DATA_PIPELINE import tokenize, tokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors)")
parser.add_argument("--prompt", type=str, default="", help="Prompt text to start generation")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0 for greedy)")
parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling cutoff")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling cutoff")
parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed config for multi-GPU inference")
args = parser.parse_args()

# 1) Load model
config = TransformerConfig()
model = TransformerModel(config)

state_dict = load_file(args.checkpoint)
model.half() 
model.load_state_dict(state_dict)
model.eval()

# 2) Initialize DeepSpeed if desired
if args.deepspeed_config:
    deepspeed.init_distributed()
    model = model.cuda()
    engine = deepspeed.initialize(model=model, config=args.deepspeed_config, model_parameters=[])[0]
else:
    engine = model.cuda() if torch.cuda.is_available() else model

# Simple decode helper for final text
def decode(token_ids):
    return tokenizer.decode(token_ids)

# 3) Prepare prompt
prompt_ids = tokenize(args.prompt)["token_ids"]
if len(prompt_ids) == 0:
    prompt_ids = []
input_ids = torch.tensor([prompt_ids], dtype=torch.long)
if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')

# If distributed, broadcast the prompt to all ranks
if hasattr(dist, "is_initialized") and dist.is_initialized():
    if dist.get_rank() == 0:
        seq_len = input_ids.size(1)
    else:
        seq_len = 0
    seq_len_t = torch.tensor([seq_len], dtype=torch.long).cuda()
    dist.broadcast(seq_len_t, src=0)
    seq_len = int(seq_len_t.item())
    if dist.get_rank() != 0:
        input_ids = torch.zeros((1, seq_len), dtype=torch.long).cuda()
    dist.broadcast(input_ids, src=0)

# 4) Generate tokens
generated_ids = input_ids.clone()
for _ in range(args.max_new_tokens):
    seq_len = generated_ids.size(1)
    if seq_len > config.block_size:
        generated_ids = generated_ids[:, -config.block_size:]
        seq_len = config.block_size
    
    with torch.no_grad():
        logits = engine(generated_ids)
    next_token_logits = logits[0, -1, :]

    # Temperature / top-k / top-p
    if args.temperature <= 0:
        # Greedy
        next_token_id = int(torch.argmax(next_token_logits).item())
    else:
        probs = F.softmax(next_token_logits / args.temperature, dim=-1)
        # Nucleus (top-p) sampling
        if args.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            cutoff_idx = torch.searchsorted(cum_probs, args.top_p)
            sorted_probs[cutoff_idx + 1:] = 0
            probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)
        # Top-k sampling
        if args.top_k > 0:
            top_k = min(args.top_k, probs.size(0))
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1]
            probs[indices_to_remove] = 0
        
        probs = probs / probs.sum()
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())
    
    # Append the new token
    next_token = torch.tensor([[next_token_id]], dtype=torch.long).cuda()
    if hasattr(dist, "is_initialized") and dist.is_initialized():
        dist.broadcast(next_token, src=0)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

# 5) Decode final output (rank 0 only)
if not hasattr(dist, "is_initialized") or not dist.is_initialized() or dist.get_rank() == 0:
    output_tokens = generated_ids[0].cpu().tolist()
    # The portion after the prompt is the "generated" text
    gen_text = decode(output_tokens[len(prompt_ids):])
    print(f"Prompt: {args.prompt}")
    print(f"Generated continuation: {gen_text}")
