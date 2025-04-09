import os
from huggingface_hub import hf_hub_download

DATA_DIR = "path"  # change if desired
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

os.makedirs(DATASETS_DIR, exist_ok=True)
def download_hf_file(repo_id, filename, subfolder=None, repo_type="dataset", revision="main"):
    print(f"Downloading {filename} from {repo_id} ...")
    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision
    )
    # Move the downloaded file from your cache to E:/Data/datasets/<subfolder>/filename
    out_dir = DATASETS_DIR
    if subfolder:
        out_dir = os.path.join(DATASETS_DIR, subfolder)
        os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, os.path.basename(filename))
    if not os.path.exists(out_path):
        os.rename(local_file, out_path)

    print(f"Saved to: {out_path}")
    return out_path

python_file = "data/python.json.gz"
download_hf_file(
    repo_id="bigcode/the-stack",
    filename=python_file,
    subfolder="the_stack_v1/python",
    revision="main"
)
owt2_file = "openwebtext2-raw.gz"
download_hf_file(
    repo_id="openwebtext2",
    filename=owt2_file,
    subfolder="openwebtext2",
    revision="main"
)

print("✅ All downloads completed.")
