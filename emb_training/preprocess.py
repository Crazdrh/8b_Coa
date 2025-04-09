import tiktoken
import os
import tarfile
import gzip
import numpy as np
from tqdm import tqdm

TOKENIZER_NAME = "o200k_base"
MAX_TOKENS_PER_SOURCE = 100_000_000  # Limit per dataset for memory
WINDOW_SIZE = 5
OUTPUT_DIR = "preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding(TOKENIZER_NAME)
vocab_size = enc.n_vocab

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
                    except Exception as e:
                        continue
    return token_ids[:MAX_TOKENS_PER_SOURCE]

def extract_and_tokenize_tsv(file_path):
    print(f"Processing TSV file: {file_path}")
    token_ids = []
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f):
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                tokens = enc.encode(fields[1])
                token_ids.extend(tokens)
                if len(token_ids) >= MAX_TOKENS_PER_SOURCE:
                    break
    return token_ids[:MAX_TOKENS_PER_SOURCE]

def make_pairs(tokens, window_size):
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        center = tokens[i]
        context_ids = tokens[i - window_size:i] + tokens[i + 1:i + 1 + window_size]
        for context in context_ids:
            pairs.append((center, context))
    return np.array(pairs, dtype=np.int32)

def save_pairs(pairs, source_name):
    out_path = os.path.join(OUTPUT_DIR, f"{source_name}_pairs.npy")
    np.save(out_path, pairs)
    print(f"Saved {len(pairs)} pairs to {out_path} ({pairs.nbytes / 1024**3:.2f} GB)")

if __name__ == "__main__":
    all_tokens = []

    sources = {
        "wiki": "datasets/wiki.tar.gz",
        "owt2": "datasets/openwebtext2.tar.gz",
        "laion": "datasets/laion2B-en.tsv.gz"
    }

    for name, path in sources.items():
        if path.endswith(".tar.gz"):
            tokens = extract_and_tokenize_tar(path)
        elif path.endswith(".tsv.gz"):
            tokens = extract_and_tokenize_tsv(path)
        else:
            continue

        pairs = make_pairs(tokens, WINDOW_SIZE)
        save_pairs(pairs, name)

    with open(os.path.join(OUTPUT_DIR, "meta.txt"), "w") as f:
        f.write(f"{vocab_size}\\n")

# Save files
multi_file_paths = []
for fname, content in multi_files.items():
    path = os.path.join(multi_data_dir, fname)
    with open(path, "w") as f:
        f.write(content)
    multi_file_paths.append(path)

# Zip the project
multi_zip_path = "/mnt/data/multi_dataset_project.zip"
with zipfile.ZipFile(multi_zip_path, "w") as zipf:
    for path in multi_file_paths:
        zipf.write(path, arcname=os.path.basename(path))

multi_zip_path