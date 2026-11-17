"""Download the ESCI parquet files into data/raw and verify their SHA-256.

Source: https://github.com/amazon-science/esci-data (Apache-2.0).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import EXAMPLES_FILE, PRODUCTS_FILE, ROOT, load_config  # noqa: E402

BASE_URL = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
# Hashes of the files as downloaded on 2026-11-17.
SHA256 = {
    EXAMPLES_FILE: "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    PRODUCTS_FILE: "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw_dir = ROOT / load_config()["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        path = raw_dir / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, path)
        actual = sha256(path)
        if actual != expected:
            sys.exit(f"{name}: sha256 {actual} does not match the pinned {expected}")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
