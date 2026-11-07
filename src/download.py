"""Download the preprocessed part of PADS from PhysioNet into data/raw/.

Only fetches what the pipeline needs (~240 MB), not the full 1.4 GB dataset:
  preprocessed/file_list.csv           labels + demographics, one row per subject
  preprocessed/movement/NNN_ml.bin     smartwatch signals, one file per subject
  preprocessed/questionnaire/NNN_ml.bin  30 non-motor symptom answers per subject

Every file is checked against PhysioNet's SHA256SUMS.txt. Re-running skips
files that are already present and valid.

Usage:  python -m src.download
"""

import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import PHYSIONET_BASE, RAW_DIR

WANTED_PREFIXES = (
    "preprocessed/file_list.csv",
    "preprocessed/movement/",
    "preprocessed/questionnaire/",
)


def fetch_checksums() -> dict[str, str]:
    with urllib.request.urlopen(f"{PHYSIONET_BASE}/SHA256SUMS.txt") as resp:
        lines = resp.read().decode().splitlines()
    checksums = {}
    for line in lines:
        digest, path = line.split(maxsplit=1)
        if path.startswith(WANTED_PREFIXES):
            checksums[path] = digest
    return checksums


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_one(rel_path: str, digest: str) -> str:
    dest = RAW_DIR / rel_path
    if dest.exists() and sha256(dest) == digest:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(f"{PHYSIONET_BASE}/{rel_path}", tmp)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {rel_path}")
    tmp.replace(dest)
    return "downloaded"


def main() -> None:
    checksums = fetch_checksums()
    print(f"{len(checksums)} files to check in {RAW_DIR}")
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one, p, d): p for p, d in checksums.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)}")
    print(f"done: {counts['downloaded']} downloaded, {counts['skipped']} already present")


if __name__ == "__main__":
    main()
