"""Shared paths and dataset constants for the PADS pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # files exactly as downloaded from PhysioNet
PROCESSED_DIR = DATA_DIR / "processed"  # our derived files (features etc.)
SPLITS_DIR = ROOT / "splits"          # committed to git so the whole group uses the same split
RESULTS_DIR = ROOT / "results"

PHYSIONET_BASE = "https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0"

RANDOM_SEED = 42
TEST_FRACTION = 0.2

# --- Classification target ---------------------------------------------------
# PADS labels (preprocessed/file_list.csv): 0 = Healthy, 1 = Parkinson's,
# 2 = other movement disorders (essential tremor, MS, atypical parkinsonism, ...).
#   "pd_vs_healthy": drop label 2, classify 1 vs 0.
#   "pd_vs_rest":    keep everyone, classify 1 vs (0 or 2). Harder, closer to real screening.
TARGET = "pd_vs_healthy"

# --- Movement data layout ----------------------------------------------------
# Each preprocessed/movement/NNN_ml.bin is float32, shape (132, 976):
# 11 tasks x 2 wrists x 2 sensors x 3 axes channels, 976 samples at 100 Hz (~9.8 s).
# Gravity is already removed from the accelerometer and the first 0.48 s is cut
# (see scripts/run_preprocessing.py in the dataset).
SAMPLING_RATE = 100
N_SAMPLES = 976
TASKS = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold",
    "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2",
]
WRISTS = ["Left", "Right"]
SENSORS = ["Acc", "Gyro"]
AXES = ["X", "Y", "Z"]
N_CHANNELS = len(TASKS) * len(WRISTS) * len(SENSORS) * len(AXES)  # 132

# Demographic column added to the feature table. Not a movement feature: only the
# sex-correction variants in train.py use it.
SEX_COL = "demo__male"

# The 30 yes/no non-motor symptom questions (PDNMS), in file order (PADS Table 1).
QUESTIONS = [
    "dribbling", "taste_smell", "swallowing", "vomiting", "constipation",
    "bowel_incontinence", "bowel_incomplete", "urinary_urgency", "nocturia", "pain",
    "weight_change", "remembering", "loss_of_interest", "hallucinations", "concentrating",
    "sad_blues", "anxiety", "sex_drive", "sex_difficulty", "dizzy",
    "falling", "daytime_sleepiness", "insomnia", "vivid_dreams", "acting_out_dreams",
    "restless_legs", "swelling", "sweating", "double_vision", "delusions",
]
