import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import deepspeed
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from model import TransformerConfig, TransformerModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data", help="Directory with train.bin and val.bin")
parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed config JSON")
parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps (iterations) to run")
parser.add_argument("--save_interval", type=int, default=1000, help="Steps between saving model checkpoints")
parser.add_argument("--eval_interval", type=int, default=1000, help="Steps between evaluating on validation set")
parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging training progress")
parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")
args = parser.parse_args()

# Initialize distributed training (DeepSpeed will handle device allocation)
deepspeed.init_distributed()

# Set each process's CUDA device based on local rank
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

# Load dataset binaries
train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint32, mode='r')
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint32, mode='r')
assert train_data.size % 2048 == 0 and val_data.size % 2048 == 0, "Data files should contain full 2048-token sequences"
train_seq_count = train_data.size // 2048
val_seq_count = val_data.size // 2048

# Initialize model config and model
config = TransformerConfig()  # uses default 200k vocab, 2048 seq_len, 32 layers, 32 heads, 4096 embed
model = TransformerModel(config)
# If resuming from a checkpoint, load weights
if args.resume:
    checkpoint_state = load_file(args.resume)
    model.load_state_dict(checkpoint_state)
# Create optimizer (AdamW with weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
# Initialize DeepSpeed engine
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
engine, optimizer, _, _ = deepspeed.initialize(
    config=args.deepspeed_config,
    model=model,
    model_parameters=model_parameters,
    optimizer=optimizer
)

# Training loop
engine.module.train()
loss_sum = 0.0
for step in range(1, args.max_steps + 1):
    # Sample a batch of random sequence indices for this micro-step
    # (Here we assume global batch size = engine.train_micro_batch_size_per_gpu * world_size * grad_accum_steps)
    # We use only local data loading per process for simplicity (each rank draws from full data).
    batch_size = engine.train_micro_batch_size_per_gpu
    seq_indices = np.random.randint(0, train_seq_count, size=batch_size)
    # Prepare batch data
    # Each sequence has length 2048; input is first 2047 tokens, target is next 2047 tokens
    X_batch = np.zeros((batch_size, 2047), dtype=np.int64)
    Y_batch = np.zeros((batch_size, 2047), dtype=np.int64)
    for i, seq_idx in enumerate(seq_indices):
        start = seq_idx * 2048
        seq = train_data[start : start + 2048]
        X_batch[i] = seq[:2047]
        Y_batch[i] = seq[1:2048]
    # Convert to torch tensors
    X_batch = torch.from_numpy(X_batch).cuda()
    Y_batch = torch.from_numpy(Y_batch).cuda()

    # Forward pass and loss computation
    logits = engine(X_batch)  # DeepSpeed engine applies model forward
    logits = logits.view(-1, config.vocab_size)
    targets = Y_batch.view(-1)
    loss = F.cross_entropy(logits.float(), targets, reduction='mean')
    # Backpropagation
    engine.backward(loss)
    engine.step()

    loss_sum += loss.item()
    # Logging
    if step % args.log_interval == 0:
        avg_loss = loss_sum / args.log_interval
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')  # avoid overflow
        if engine.global_rank == 0:
            print(f"Step {step}: avg_train_loss={avg_loss:.4f}, perplexity={ppl:.2f}")
        loss_sum = 0.0

    # Evaluation on validation set
    if args.eval_interval > 0 and step % args.eval_interval == 0:
        engine.module.eval()
        # Each rank evaluates a subset of validation sequences
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        seq_per_rank = math.ceil(val_seq_count / world_size)
        start_seq = rank * seq_per_rank
        end_seq = min(val_seq_count, (rank + 1) * seq_per_rank)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for seq_idx in range(start_seq, end_seq):
                start = seq_idx * 2048
                seq = torch.from_numpy(val_data[start : start + 2048].astype(np.int64)).cuda()
                # Input and target
                input_ids = seq[:2047].unsqueeze(0)  # (1, 2047)
                targets = seq[1:2048].unsqueeze(0)    # (1, 2047)
                # Forward
                outputs = engine(input_ids)  # (1, 2047, vocab_size)
                # Compute loss sum for this sequence
                loss_val = F.cross_entropy(outputs.view(-1, config.vocab_size).float(), targets.view(-1), reduction='sum')
                total_loss += loss_val.item()
                total_tokens += targets.numel()
        # Gather total loss and token count from all ranks
        total_loss_tensor = torch.tensor([total_loss], dtype=torch.float64).cuda()
        total_tokens_tensor = torch.tensor([total_tokens], dtype=torch.float64).cuda()
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        if engine.global_rank == 0:
            mean_loss = total_loss_tensor.item() / total_tokens_tensor.item()
            val_ppl = math.exp(mean_loss)
            print(f"Step {step}: validation perplexity = {val_ppl:.3f}")
        engine.module.train()

    # Save checkpoint periodically
    if args.save_interval > 0 and step % args.save_interval == 0:
        if engine.global_rank == 0:
            # Consolidate model weights and save to safetensors
            state_dict = engine.module.state_dict()
            # Move weights to CPU to save memory during save
            for k, v in state_dict.items():
                state_dict[k] = v.to(device='cpu')
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{step}.safetensors")
            os.makedirs(args.out_dir, exist_ok=True)
            save_file(state_dict, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")