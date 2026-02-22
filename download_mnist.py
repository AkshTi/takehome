#!/usr/bin/env python3
"""
MNIST Download Script
Downloads the MNIST dataset and stores it in the ./data folder.
"""

import os
import urllib.request
import gzip
import shutil

# Create data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# MNIST dataset URLs (Yann LeCun's original host mirror via ossci-datasets)
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_and_extract(filename: str, dest_dir: str) -> None:
    url = BASE_URL + filename
    gz_path = os.path.join(dest_dir, filename)
    extracted_path = os.path.join(dest_dir, filename.replace(".gz", ""))

    if os.path.exists(extracted_path):
        print(f"  [skip] {filename.replace('.gz', '')} already exists.")
        return

    # Download
    print(f"  Downloading {filename} ...")
    urllib.request.urlretrieve(url, gz_path)

    # Extract
    print(f"  Extracting {filename} ...")
    with gzip.open(gz_path, "rb") as f_in, open(extracted_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Remove the .gz file
    os.remove(gz_path)
    print(f"  Done -> {extracted_path}")


def main():
    print(f"Saving MNIST data to: {DATA_DIR}\n")
    for fname in FILES:
        download_and_extract(fname, DATA_DIR)

    print("\nAll MNIST files downloaded successfully!")
    print("\nFiles in data/:")
    for f in sorted(os.listdir(DATA_DIR)):
        size_mb = os.path.getsize(os.path.join(DATA_DIR, f)) / (1024 * 1024)
        print(f"  {f:<40} {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
