"""Read the downloaded PADS files into numpy / pandas."""

import numpy as np
import pandas as pd

from src.config import AXES, N_CHANNELS, N_SAMPLES, RAW_DIR, SENSORS, TARGET, TASKS, WRISTS


def load_subjects() -> pd.DataFrame:
    """One row per subject with demographics and a binary `y` for the chosen TARGET.

    `id` is kept as a zero-padded string ("001") because that is how files are named.
    """
    df = pd.read_csv(RAW_DIR / "preprocessed" / "file_list.csv", dtype={"id": str})
    if TARGET == "pd_vs_healthy":
        df = df[df["label"].isin([0, 1])].copy()
    elif TARGET != "pd_vs_rest":
        raise ValueError(f"unknown TARGET {TARGET!r}")
    df["y"] = (df["label"] == 1).astype(int)
    return df.reset_index(drop=True)


def load_movement(subject_id: str) -> np.ndarray:
    """Signals for one subject, shape (task, wrist, sensor, axis, time) = (11, 2, 2, 3, 976).

    The file's channel order is task -> wrist -> sensor -> axis, so a plain reshape works.
    """
    path = RAW_DIR / "preprocessed" / "movement" / f"{subject_id}_ml.bin"
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size != N_CHANNELS * N_SAMPLES:
        raise ValueError(f"{path.name}: expected {N_CHANNELS * N_SAMPLES} values, got {flat.size}")
    return flat.reshape(len(TASKS), len(WRISTS), len(SENSORS), len(AXES), N_SAMPLES)


def load_questionnaire(subject_id: str) -> np.ndarray:
    """30 yes/no non-motor symptom answers (as 0.0 / 1.0)."""
    path = RAW_DIR / "preprocessed" / "questionnaire" / f"{subject_id}_ml.bin"
    return np.fromfile(path, dtype=np.float32)
