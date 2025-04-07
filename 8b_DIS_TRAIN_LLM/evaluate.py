import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import deepspeed
import torch.distributed as dist
from safetensors.torch import load_file
from model import TransformerConfig, TransformerModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.safetensors)")
parser.add_argument("--data_file", type=str, required=True, help="Path to tokenized data file (.bin) for evaluation")
parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed config for multi-GPU evaluation")
args = parser.parse_args()

# Initialize distributed inference if needed
if args.deepspeed_config:
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

# Load model
config = TransformerConfig()
model = TransformerModel(config)
state_dict = load_file(args.checkpoint)
model.half()  # model was trained in fp16
model.load_state_dict(state_dict)
model.eval()
# Initialize DeepSpeed engine if applicable
if args.deepspeed_config:
    model = model.cuda()
    engine = deepspeed.initialize(model=model, config=args.deepspeed_config, model_parameters=[])[0]
else:
    engine = model.cuda() if torch.cuda.is_available() else model

# Load evaluation data
data = np.memmap(args.data_file, dtype=np.uint32, mode='r')
seq_len = config.block_size
total_tokens = data.shape[0] - (data.shape[0] % seq_len)
total_seq = total_tokens // seq_len

# Distributed splitting of evaluation data
if hasattr(dist, "is_initialized") and dist.is_initialized():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0
seq_per_rank = math.ceil(total_seq / world_size)
start_seq = rank * seq_per_rank
end_seq = min(total_seq, (rank + 1) * seq_per_rank)

# Evaluate perplexity
sum_loss = 0.0
count = 0
with torch.no_grad():
    for seq_idx in range(start_seq, end_seq):
        start = seq_idx * seq_len
        seq = torch.from_numpy(data[start : start + seq_len].astype(np.int64))
        if torch.cuda.is_available():
            seq = seq.cuda()
        # Use first 2047 tokens to predict next 2047 tokens
        input_ids = seq[: seq_len-1].unsqueeze(0)   # shape (1, 2047)
        targets = seq[1: seq_len].unsqueeze(0)      # shape (1, 2047)
        outputs = engine(input_ids)                 # (1, 2047, vocab_size)
        # Compute total negative log-likelihood for this sequence
        loss = F.cross_entropy(outputs.view(-1, config.vocab_size).float(), targets.view(-1), reduction='sum')
        sum_loss += loss.item()
        count += targets.numel()
# Sum across ranks
sum_loss_tensor = torch.tensor([sum_loss], dtype=torch.double).cuda() if torch.cuda.is_available() else torch.tensor([sum_loss], dtype=torch.double)
count_tensor = torch.tensor([count], dtype=torch.double).cuda() if torch.cuda.is_available() else torch.tensor([count], dtype=torch.double)
if hasattr(dist, "is_initialized") and dist.is_initialized():
    dist.all_reduce(sum_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
if rank == 0:
    mean_nll = sum_loss_tensor.item() / count_tensor.item()
    ppl = math.exp(mean_nll)
    print(f"Perplexity = {ppl:.3f}")