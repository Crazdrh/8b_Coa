import tiktoken
import os
import tarfile
import gzip
import json
import numpy as np
from tqdm import tqdm
import zipfile

TOKENIZER_NAME = "o200k_base"
MAX_TOKENS_PER_SOURCE = 100_000_000
WINDOW_SIZE = 5
OUTPUT_DIR = "preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding(TOKENIZER_NAME)
vocab_size = enc.n_vocab

def extract_and_tokenize_json_gz(file_path):
    print(f"Processing JSON gzip file: {file_path}")
    token_ids = []

    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f):
            try:
                record = json.loads(line)
                text = record.get("content", "")  # adjust key if needed
                tokens = enc.encode(text)
                token_ids.extend(tokens)

                if len(token_ids) >= MAX_TOKENS_PER_SOURCE:
                    break
            except json.JSONDecodeError:
                continue

    return token_ids[:MAX_TOKENS_PER_SOURCE]


def extract_and_tokenize_tar(file_path):
    print(f"Processing tar file: {file_path}")
    token_ids = []

    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tqdm(tar.getmembers()):
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    try:
                        text = f.read().decode('utf-8', errors='ignore')
                        tokens = enc.encode(text)
                        token_ids.extend(tokens)
                        if len(token_ids) >= MAX_TOKENS_PER_SOURCE:
                            break
                    except Exception:
                        continue
    return token_ids[:MAX_TOKENS_PER_SOURCE]

def make_pairs(tokens, window_size):
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        center = tokens[i]
        context_ids = tokens[i - window_size: i] + tokens[i + 1: i + 1 + window_size]
        for context in context_ids:
            pairs.append((center, context))
    return np.array(pairs, dtype=np.int32)


def save_tokens(tokens, source_name):
    tokens_path = os.path.join(OUTPUT_DIR, f"{source_name}_tokens.npy")
    tokens_array = np.array(tokens, dtype=np.int32)
    np.save(tokens_path, tokens_array)
    gb_size = tokens_array.nbytes / 1024 ** 3
    print(f"Saved {len(tokens_array)} tokens to {tokens_path} ({gb_size:.2f} GB)")


def save_pairs(pairs, source_name):
    out_path = os.path.join(OUTPUT_DIR, f"{source_name}_pairs.npy")
    np.save(out_path, pairs)
    gb_size = pairs.nbytes / 1024 ** 3
    print(f"Saved {len(pairs)} pairs to {out_path} ({gb_size:.2f} GB)")

if __name__ == "__main__":
    # Example: adjust these to your actual data
    sources = {
        "owt2": "E:/Data/datasets/openwebtext2.tar.gz"
    }

    for name, path in sources.items():
        if path.endswith(".json.gz"):
            tokens = extract_and_tokenize_json_gz(path)
        elif path.endswith(".tar.gz"):
            tokens = extract_and_tokenize_tar(path)
        else:
            print(f"Skipping {path}, unsupported file type.")
            continue

        save_tokens(tokens, name)
        pairs = make_pairs(tokens, WINDOW_SIZE)
        save_pairs(pairs, name)

    # Save vocab size or other metadata
    with open(os.path.join(OUTPUT_DIR, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"{vocab_size}\n")

    print("✅ Preprocessing complete.")

    multi_files = {
        "readme.txt": "This is a readme for the multi-dataset project.\n",
    }
    multi_data_dir = "/mnt/data"  # change if desired
    os.makedirs(multi_data_dir, exist_ok=True)

    # Write each file to disk
    multi_file_paths = []
    for fname, content in multi_files.items():
        path = os.path.join(multi_data_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        multi_file_paths.append(path)

    # Now create the zip archive
    multi_zip_path = "/mnt/data/multi_dataset_project.zip"
    with zipfile.ZipFile(multi_zip_path, "w") as zipf:
        for path in multi_file_paths:
            zipf.write(path, arcname=os.path.basename(path))

    print(f"All extra files zipped at: {multi_zip_path}")
