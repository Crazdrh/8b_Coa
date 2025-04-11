#!/usr/bin/env python3

import os
import subprocess
import urllib.request
import gzip

BASE_DIR = "/home/hayden/8b_LLM/8b_Coa/tokenized_data/"
os.makedirs(BASE_DIR, exist_ok=True)
crawl_ids = [
    "CC-MAIN-2023-06",
    "CC-MAIN-2025-13",
    "CC-MAIN-2025-08",
    "CC-MAIN-2025-05"
]

def download_wet_files_for_crawl(crawl_id):

    wet_paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/wet.paths.gz"
    
    print(f"Downloading WET paths from {wet_paths_url} ...")
    
    local_wet_paths_gz = os.path.join(BASE_DIR, f"{crawl_id}-wet.paths.gz")
    local_wet_paths = os.path.join(BASE_DIR, f"{crawl_id}-wet.paths")

    urllib.request.urlretrieve(wet_paths_url, local_wet_paths_gz)

    with gzip.open(local_wet_paths_gz, 'rb') as f_in:
        with open(local_wet_paths, 'wb') as f_out:
            f_out.write(f_in.read())

    os.remove(local_wet_paths_gz)
    
    print(f"Extracted paths file to {local_wet_paths}")

    crawl_dir = os.path.join(BASE_DIR, crawl_id)
    os.makedirs(crawl_dir, exist_ok=True)

    with open(local_wet_paths, 'r') as f:
        wet_files = [line.strip() for line in f if line.strip()]

    print(f"Starting download of {len(wet_files)} WET files for {crawl_id}.")

    for i, wet_file_path in enumerate(wet_files, 1):
        wet_file_url = f"https://data.commoncrawl.org/{wet_file_path}"

        cmd = [
            "wget",
            "-c",
            "-q",
            "-P", crawl_dir,
            wet_file_url
        ]
        if i % 50 == 0:
            print(f"  Downloaded {i} / {len(wet_files)} from {crawl_id} ...")
        
        subprocess.run(cmd, check=True)

    print(f"Finished downloading WET files for {crawl_id}.\n")

def main():
    for crawl_id in crawl_ids:
        download_wet_files_for_crawl(crawl_id)

if __name__ == "__main__":
    main()
