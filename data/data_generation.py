#!/usr/bin/env python3
"""
data/data_generation.py — Download and extract the Gowalla dataset.

Downloads loc-gowalla_totalCheckins.txt.gz from the Stanford SNAP repository,
extracts it, and saves it as data/loc-gowalla_totalCheckins.txt.

Expected output
---------------
data/loc-gowalla_totalCheckins.txt
    Tab-separated file with columns:
        user | check-in_time | latitude | longitude | location_id

    Source: https://snap.stanford.edu/data/loc-Gowalla.html
"""

import gzip
import os
import shutil
import urllib.request

DOWNLOAD_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GZ_FILE      = os.path.join(SCRIPT_DIR, "loc-gowalla_totalCheckins.txt.gz")
OUT_FILE     = os.path.join(SCRIPT_DIR, "loc-gowalla_totalCheckins.txt")


def download(url: str, dest: str) -> None:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        while True:
            block = response.read(chunk)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            if total:
                print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="\r")
    print()


def extract(gz_path: str, out_path: str) -> None:
    print(f"Extracting {gz_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if os.path.exists(OUT_FILE):
        print(f"Dataset already exists at {OUT_FILE} — nothing to do.")
    else:
        download(DOWNLOAD_URL, GZ_FILE)
        extract(GZ_FILE, OUT_FILE)
        os.remove(GZ_FILE)
        print("Done.")