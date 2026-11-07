"""Make the train/test split once, at subject level, and save it to splits/.

Each subject appears in exactly one of the two files, so no person's data can
leak from training into testing. Stratified, so both sides keep the PD/healthy ratio.

The split files are committed to git: everyone in the group uses the same test
set, and nobody should look at test subjects while exploring or tuning.
Re-running refuses to overwrite an existing split unless --force is given.

Usage:  python -m src.split [--force]
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, SPLITS_DIR, TARGET, TEST_FRACTION
from src.load import load_subjects


def split_paths():
    return SPLITS_DIR / f"{TARGET}_train_ids.txt", SPLITS_DIR / f"{TARGET}_test_ids.txt"


def load_split() -> tuple[list[str], list[str]]:
    train_path, test_path = split_paths()
    return train_path.read_text().split(), test_path.read_text().split()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args()

    train_path, test_path = split_paths()
    if train_path.exists() and not args.force:
        raise SystemExit(f"{train_path} already exists; use --force to overwrite (changes the test set!)")

    subjects = load_subjects()
    train_ids, test_ids = train_test_split(
        subjects["id"],
        test_size=TEST_FRACTION,
        stratify=subjects["y"],
        random_state=RANDOM_SEED,
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(sorted(train_ids)) + "\n")
    test_path.write_text("\n".join(sorted(test_ids)) + "\n")

    y = subjects.set_index("id")["y"]
    for name, ids in [("train", train_ids), ("test", test_ids)]:
        counts = y[ids].value_counts().reindex([1, 0], fill_value=0)
        print(f"{name}: {len(ids)} subjects ({counts[1]} PD, {counts[0]} non-PD)")


if __name__ == "__main__":
    main()
