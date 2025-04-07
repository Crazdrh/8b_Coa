# 8B Transformer Language Model Training Suite

**Hardware Assumptions:** This setup targets a single node with 9 GPUs (e.g., 9× NVIDIA Titan X 12GB). GPU memory is augmented by DeepSpeed ZeRO-3 CPU offloading to handle the 8B model. Ensure you have sufficient CPU RAM and high-speed storage for dataset preprocessing and checkpoints.

## Environment and Installation

Install the required libraries and frameworks:

- **Python 3.8+** with **PyTorch 2.x** and CUDA support.
- **DeepSpeed** (for distributed training). Install via pip: `pip install deepspeed`.
- **Hugging Face Datasets** (for data loading): `pip install datasets`.
- **tiktoken** (OpenAI BPE tokenizer): `pip install tiktoken`.
- **safetensors** (for safe checkpoint saving): `pip install safetensors`.
- Ensure that **NVIDIA Apex** or other dependencies for DeepSpeed are installed if needed (DeepSpeed might install its own fused optimizers).
- This code assumes a Linux environment with CUDA drivers properly configured for the GPUs.

## Data: C4, OpenWebText, and The Pile

We will train on a combination of **C4 (Colossal Cleaned Common Crawl, English subset)**, **OpenWebText**, and **The Pile** datasets. These are large-scale text corpora:
- **C4 (English)**: ~806 GB of text (about 156 billion tokens)&#8203;:contentReference[oaicite:0]{index=0}.
- **OpenWebText**: ~38 GB of text extracted from Reddit links (about 8 million documents)&#8203;:contentReference[oaicite:1]{index=1}.
- **The Pile**: ~825 GiB diverse text from 22 sources (academic, internet forums, etc.)&#8203;:contentReference[oaicite:2]{index=2}.

**Expected Disk Usage:** Full datasets combined are over **1.6 TB** of raw text. Tokenized data stored in binary form (4 bytes per token) can also exceed **1 TB**. Ensure you have ample disk space (and consider using high-speed SSD/NVMe for faster preprocessing). If storage or bandwidth is a concern, you may choose to use a subset or filtered portion of these datasets.

## Preprocessing the Dataset

The `preprocess_dataset.py` script downloads (streams) the datasets, tokenizes the text, and writes out fixed-length sequences of tokens (2048 tokens each) to binary files. By default, it also sets aside a small fraction of data as a validation set for perplexity evaluation.

**Steps:**
1. **Set Up Tokenizer:** We use OpenAI's `tiktoken` implementation of the GPT-2 BPE tokenizer. The vocabulary size is expanded to **200,000 tokens** for training (to accommodate diverse text). In our code, we use the GPT-2 encoding as a base. The special end-of-text token `<|endoftext|>` is used to separate documents.
2. **Run Preprocessing:** Launch the script:
   ```bash
   python preprocess_dataset.py --output_dir data/ --val_fraction 0.001 --seed 0
