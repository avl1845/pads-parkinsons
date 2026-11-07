"""Turn raw smartwatch signals into one row of hand-crafted features per subject.

Columns are prefixed with their feature group, so train.py can pick groups:

  base__   per (task, wrist, sensor), on the 3-axis magnitude signal:
           rms, jerk_rms, dom_freq, tremor_power_rel, spectral_entropy.
           11 tasks x 2 wrists x 2 sensors x 5 = 220 columns.

  trem__   tremor features from the AXIS-SUMMED spectrum (welch per X/Y/Z, then
           added). Unlike the magnitude signal this does not depend on how the
           watch is oriented and does not fold a 5 Hz tremor up to ~10 Hz
           (taking |x| of an oscillation doubles its frequency).
           Per (task, wrist, sensor): power, rel, peak_freq, sharpness  -> 176
           Plus: distraction effect (resting tremor while counting serial sevens
           vs plain resting, per wrist x sensor) -> 4, and late-vs-early tremor
           while holding the arms out (re-emergent tremor) -> 4.

  asym__   left/right asymmetry |log(left) - log(right)| for positive-valued
           base and tremor features: 0 = symmetric, log(2) = one side twice the
           other. PD usually starts on one side. 11 tasks x 2 sensors x 7 = 154.

  quest__  the 30 yes/no non-motor symptom answers (1 = yes), e.g. constipation,
           loss of smell, acting out dreams. Not watch data: the "app questionnaire".

  demo__male  1 = male, 0 = female. Not a movement feature; only used by the
           sex-correction variants in train.py.

Ideas not yet included: bradykinesia/smoothness of the
kinetic tasks, per-axis features.

Usage:  python -m src.features      -> data/processed/features.csv
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

from src.config import PROCESSED_DIR, QUESTIONS, SAMPLING_RATE, SENSORS, SEX_COL, TASKS, WRISTS
from src.load import load_movement, load_questionnaire, load_subjects

TREMOR_BAND = (3.0, 7.0)  # Hz, typical Parkinsonian rest tremor (4-6 Hz) with margin
BROAD_BAND = (1.0, 15.0)  # Hz, range of voluntary + tremor movement
NPERSEG = 256             # welch window: 2.56 s, ~0.4 Hz resolution
EPS = 1e-12

RESTING = ["Relaxed1", "Relaxed2"]                # task 1a: resting, arms on lap
RESTING_DISTRACTED = ["RelaxedTask1", "RelaxedTask2"]  # task 1b: resting + serial sevens
HOLD_TASK = "StretchHold"                         # arms held out


def band(freqs, lo_hi) -> np.ndarray:
    return (freqs >= lo_hi[0]) & (freqs <= lo_hi[1])


def magnitude_features(sig: np.ndarray) -> dict[str, float]:
    """Features of one 1-D magnitude signal sampled at SAMPLING_RATE."""
    sig = sig - sig.mean()
    freqs, psd = welch(sig, fs=SAMPLING_RATE, nperseg=NPERSEG)
    total_power = psd.sum() + EPS
    jerk = np.diff(sig) * SAMPLING_RATE
    return {
        # mean already removed, so RMS == standard deviation; keep only one
        "rms": float(np.sqrt(np.mean(sig**2))),
        "jerk_rms": float(np.sqrt(np.mean(jerk**2))),
        "dom_freq": float(freqs[np.argmax(psd[1:]) + 1]),  # skip the DC bin
        "tremor_power_rel": float(psd[band(freqs, TREMOR_BAND)].sum() / total_power),
        "spectral_entropy": float(entropy(psd / total_power)),
    }


def axis_summed_psd(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a (3, time) signal, summed over axes (orientation-invariant)."""
    freqs, psd = welch(xyz, fs=SAMPLING_RATE, nperseg=NPERSEG, axis=-1)  # removes mean per segment
    return freqs, psd.sum(axis=0)


def tremor_band_power(xyz: np.ndarray) -> float:
    freqs, psd = axis_summed_psd(xyz)
    return float(psd[band(freqs, TREMOR_BAND)].sum())


def tremor_features(xyz: np.ndarray) -> dict[str, float]:
    """Tremor features of one (3, time) sensor recording. `power` is raw (log-transformed later)."""
    freqs, psd = axis_summed_psd(xyz)
    in_band, in_broad = band(freqs, TREMOR_BAND), band(freqs, BROAD_BAND)
    power = psd[in_band].sum()
    return {
        "power": float(power),                                    # how strong the tremor is
        "rel": float(power / (psd[in_broad].sum() + EPS)),        # tremor share of all movement
        "peak_freq": float(freqs[in_broad][np.argmax(psd[in_broad])]),  # main rhythm, Hz
        # how much the tremor peak sticks out: real tremor = narrow spike, fidgeting = flat
        "sharpness": float(np.log10(psd[in_band].max() / (np.median(psd[in_broad]) + EPS) + EPS)),
    }


def log_ratio(a: float, b: float) -> float:
    return float(np.log10((a + EPS) / (b + EPS)))


def subject_features(subject_id: str) -> dict[str, float]:
    signals = load_movement(subject_id)  # (task, wrist, sensor, axis, time)
    base, trem = {}, {}
    for t, task in enumerate(TASKS):
        for w, wrist in enumerate(WRISTS):
            for s, sensor in enumerate(SENSORS):
                xyz = signals[t, w, s]
                base[task, wrist, sensor] = magnitude_features(np.linalg.norm(xyz, axis=0))
                trem[task, wrist, sensor] = tremor_features(xyz)

    row = {}
    for (task, wrist, sensor), feats in base.items():
        for name, value in feats.items():
            row[f"base__{task}_{wrist}_{sensor}_{name}"] = value
    for (task, wrist, sensor), feats in trem.items():
        for name, value in feats.items():
            value = np.log10(value + EPS) if name == "power" else value  # power is very skewed
            row[f"trem__{task}_{wrist}_{sensor}_{name}"] = value

    ti = {task: i for i, task in enumerate(TASKS)}
    for w, wrist in enumerate(WRISTS):
        for s, sensor in enumerate(SENSORS):
            # Distraction: PD rest tremor typically gets worse during mental effort.
            rest = sum(trem[task, wrist, sensor]["power"] for task in RESTING)
            busy = sum(trem[task, wrist, sensor]["power"] for task in RESTING_DISTRACTED)
            row[f"trem__distraction_{wrist}_{sensor}"] = log_ratio(busy, rest)
            # Re-emergent tremor: PD tremor often returns a few seconds into a held posture.
            xyz = signals[ti[HOLD_TASK], w, s]
            half = xyz.shape[-1] // 2
            row[f"trem__hold_late_vs_early_{wrist}_{sensor}"] = log_ratio(
                tremor_band_power(xyz[:, half:]), tremor_band_power(xyz[:, :half]))

    asym_sources = [(base, ["rms", "jerk_rms", "dom_freq", "tremor_power_rel", "spectral_entropy"]),
                    (trem, ["power", "rel"])]
    for source, names in asym_sources:
        prefix = "" if source is base else "trem_"
        for task in TASKS:
            for sensor in SENSORS:
                left, right = source[task, "Left", sensor], source[task, "Right", sensor]
                for name in names:
                    row[f"asym__{task}_{sensor}_{prefix}{name}"] = abs(log_ratio(left[name], right[name]))
    return row


def build_feature_table() -> pd.DataFrame:
    subjects = load_subjects()
    rows = [subject_features(sid) for sid in subjects["id"]]
    feats = pd.DataFrame(rows)
    feats.insert(0, "id", subjects["id"].values)
    feats.insert(1, "y", subjects["y"].values)
    answers = np.array([load_questionnaire(sid) for sid in subjects["id"]])
    quest = pd.DataFrame(answers, columns=[f"quest__q{i:02d}_{q}" for i, q in enumerate(QUESTIONS, 1)])
    feats = pd.concat([feats, quest], axis=1)
    feats[SEX_COL] = (subjects["gender"] == "male").astype(int).values
    return feats


def main() -> None:
    feats = build_feature_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features.csv"
    feats.to_csv(out, index=False)
    groups = feats.columns[2:].str.split("__").str[0].value_counts().to_dict()
    print(f"wrote {out}: {len(feats)} subjects x {feats.shape[1] - 2} features {groups}")


if __name__ == "__main__":
    main()
