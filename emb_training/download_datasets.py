import os
import requests

DATA_DIR = r"E:/Data"
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def download_file(url, out_path):
    print(f"Downloading from {url} to {out_path} ...")
    # Disable certificate verification here:
    response = requests.get(url, stream=True, verify=False)  
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# === Dataset 1: The Pile (Subset for demo) ===
PILE_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/wiki.tar.gz"
download_file(PILE_URL, os.path.join(DATASETS_DIR, "wiki.tar.gz"))

# === Dataset 2: OpenWebText2 ===
OWT2_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.tar.gz"
download_file(OWT2_URL, os.path.join(DATASETS_DIR, "openwebtext2.tar.gz"))

# === Dataset 3: LAION2B-en captions subset ===
LAION_URL = "https://huggingface.co/datasets/laion/laion2B-en/resolve/main/laion2B-en.tsv.gz"
download_file(LAION_URL, os.path.join(DATASETS_DIR, "laion2B-en.tsv.gz"))

print("✅ Downloads completed (with verify=False).")
