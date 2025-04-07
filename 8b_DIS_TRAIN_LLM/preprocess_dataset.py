import os
import random
import numpy as np
from datasets import load_dataset
from DATA_PIPELINE import tokenize

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output .bin files")
parser.add_argument("--val_fraction", type=float, default=0.001, help="Fraction of data to use for validation")
parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling validation data")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
train_bin_path = os.path.join(args.output_dir, "train.bin")
val_bin_path = os.path.join(args.output_dir, "val.bin")

train_out = open(train_bin_path, "wb")
val_out = open(val_bin_path, "wb")

train_buffer = []
val_buffer = []
train_seq_count = 0
val_seq_count = 0

random.seed(args.seed)

c4_train = load_dataset("allenai/c4", "en", split="train", streaming=True)
c4_val   = load_dataset("allenai/c4", "en", split="validation", streaming=True)
owt_train = load_dataset("openwebtext", split="train", streaming=True)
pile_train = load_dataset("the_pile", split="train", streaming=True)

def flush_buffer(buffer, out_file, seq_len=2048):
    """
    Writes chunks of 'seq_len' tokens from 'buffer' into 'out_file'
    as 32-bit integers. Once a chunk is written, it is removed from 'buffer'.
    """
    global train_seq_count, val_seq_count
    while len(buffer) >= seq_len:
        chunk = np.array(buffer[:seq_len], dtype=np.uint32)
        chunk.tofile(out_file)
        buffer[:] = buffer[seq_len:]
        if out_file is train_out:
            train_seq_count += 1
        else:
            val_seq_count += 1


print("Processing C4 validation split...")
for example in c4_val:
    text = example["text"]
    ids = tokenize(text)["token_ids"]
    val_buffer.extend(ids)
    flush_buffer(val_buffer, val_out)

print("Streaming training data from C4, OpenWebText, and The Pile...")
datasets = [c4_train, owt_train, pile_train]
iters = [iter(ds) for ds in datasets]
finished = 0

while finished < len(iters):
    for i, it in enumerate(iters):
        if it is None:
            continue
        try:
            example = next(it)
        except StopIteration:
            iters[i] = None
            finished += 1
            continue
        text = example["text"] if "text" in example else str(example)
        ids = tokenize(text)["token_ids"]
        if random.random() < args.val_fraction:
            val_buffer.extend(ids)
            flush_buffer(val_buffer, val_out)
        else:
            train_buffer.extend(ids)
            flush_buffer(train_buffer, train_out)

train_buffer = []
val_buffer   = []
train_out.close()
val_out.close()

print(f"Finished preprocessing. Saved training sequences to {train_bin_path} and validation sequences to {val_bin_path}.")
print(f"Total training sequences (of 2048 tokens each): {train_seq_count}")
print(f"Total validation sequences (of 2048 tokens each): {val_seq_count}")
