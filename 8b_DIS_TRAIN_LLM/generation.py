import os
import torch
import torch.nn.functional as F
import deepspeed
import torch.distributed as dist
from safetensors.torch import load_file
from model import TransformerConfig, TransformerModel
from DATA_PIPELINE import encode_text, decode_tokens, EOS_TOKEN, EOS_ID

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

# Initialize model
config = TransformerConfig()  # same config as training
model = TransformerModel(config)

# Load checkpoint weights
state_dict = load_file(args.checkpoint)
# If checkpoint is in FP16 and model initialized in FP32, convert model to FP16 first
# (Assume we saved in FP16 from training)
model.half()
model.load_state_dict(state_dict)
model.eval()

# If using DeepSpeed for inference across GPUs
if args.deepspeed_config:
    deepspeed.init_distributed()
    model = model.cuda()  # move initial model to GPU(s)
    engine = deepspeed.initialize(model=model, config=args.deepspeed_config, model_parameters=[])[0]
else:
    # Single GPU or CPU inference
    engine = model.cuda() if torch.cuda.is_available() else model

# Prepare prompt tokens
prompt = args.prompt
prompt_ids = encode_text(prompt)
if len(prompt_ids) == 0:
    prompt_ids = []  # empty prompt
input_ids = torch.tensor([prompt_ids], dtype=torch.long)
# Move to GPU if available
input_ids = input_ids.to('cuda') if torch.cuda.is_available() else input_ids

# If distributed, ensure each rank has the same input
if hasattr(dist, "is_initialized") and dist.is_initialized():
    # Broadcast initial prompt from rank 0 to all ranks
    if dist.get_rank() == 0:
        seq_len = input_ids.size(1)
    else:
        seq_len = 0
    # Broadcast sequence length first
    seq_len = torch.tensor([seq_len], dtype=torch.long).cuda()
    dist.broadcast(seq_len, src=0)
    seq_len = int(seq_len.item())
    if dist.get_rank() != 0:
        # create empty input_ids on other ranks and receive prompt
        input_ids = torch.zeros((1, seq_len), dtype=torch.long).cuda()
    dist.broadcast(input_ids, src=0)

# Generation loop
generated_ids = input_ids.clone()
for _ in range(args.max_new_tokens):
    seq_len = generated_ids.size(1)
    # If sequence is longer than block size, truncate the oldest tokens
    if seq_len > config.block_size:
        generated_ids = generated_ids[:, -config.block_size:]
        seq_len = config.block_size
    # Get model output for the current sequence
    with torch.no_grad():
        logits = engine(generated_ids)
    # Focus on the last token's logits
    next_token_logits = logits[0, -1, :]  # (vocab_size,)
    # Apply temperature
    if args.temperature is None or args.temperature <= 0:
        # Greedy if temperature is 0 or negative
        next_token_id = int(torch.argmax(next_token_logits).item())
    else:
        probs = F.softmax(next_token_logits / args.temperature, dim=-1)
        # Top-p (nucleus) filtering
        if args.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # Mask tokens beyond top-p cutoff
            sorted_cutoff_index = torch.searchsorted(cumulative_probs, args.top_p)
            sorted_probs[sorted_cutoff_index + 1:] = 0
            probs = torch.zeros_like(probs).scatter(0, sorted_indices, sorted_probs)
        # Top-k filtering
        if args.top_k > 0:
            top_k = min(args.top_k, probs.size(0))
            indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1]
            probs[indices_to_remove] = 0
        # Re-normalize after filtering
        probs = probs / probs.sum()
        # Sample from the distribution
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())
    # Append the sampled token
    next_token = torch.tensor([[next_token_id]], dtype=torch.long)
    if hasattr(dist, "is_initialized") and dist.is_initialized():
        # Broadcast the chosen token to all ranks
        dist.broadcast(next_token.cuda(), src=0)
    # Concatenate to sequence
    generated_ids = torch.cat((generated_ids, next_token.cuda()), dim=1)
    # Stop if EOS token is generated
    if next_token_id == EOS_ID:
        break

# Decode the generated tokens to text (only do this on rank 0)
if not hasattr(dist, "is_initialized") or not dist.is_initialized() or dist.get_rank() == 0:
    output_tokens = generated_ids.cpu().tolist()[0]
    # Separate prompt and new completion
    prompt_len = len(prompt_ids)
    generated_tokens = output_tokens[prompt_len:]
    generated_text = decode_tokens(generated_tokens)
    print(f"Prompt: {prompt}")
    print(f"Generated continuation: {generated_text}")