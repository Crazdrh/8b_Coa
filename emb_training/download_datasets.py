import os
import subprocess

os.makedirs("datasets", exist_ok=True)

def download_file(url, out_path):
    print(f"Downloading from {url} to {out_path} ...")
    subprocess.run(["wget", "-c", url, "-O", out_path], check=True)

# === Dataset 1: The Pile (Subset for demo) ===
PILE_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/wiki.tar.gz"
download_file(PILE_URL, "datasets/wiki.tar.gz")

# === Dataset 2: OpenWebText2 ===
OWT2_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.tar.gz"
download_file(OWT2_URL, "datasets/openwebtext2.tar.gz")

# === Dataset 3: LAION2B-en captions subset ===
LAION_URL = "https://huggingface.co/datasets/laion/laion2B-en/resolve/main/laion2B-en.tsv.gz"
download_file(LAION_URL, "datasets/laion2B-en.tsv.gz")

# Optional: add more datasets as needed
print("✅ Downloads completed.")