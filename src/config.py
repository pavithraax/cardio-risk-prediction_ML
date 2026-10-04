"""Central configuration: paths, constants, feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "framingham.csv"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
METRICS_PATH = ROOT / "results" / "metrics.csv"

DATA_URL = ("https://raw.githubusercontent.com/GauravPadawe/"
            "Framingham-Heart-Study/master/framingham.csv")

TARGET = "TenYearCHD"
SEED = 42
TEST_SIZE = 0.20      # 80/20 split, as in the Stanford paper
CV_FOLDS = 10         # 10-fold CV, as in the Stanford paper

BASE_FEATURES = ["male", "age", "education", "currentSmoker", "cigsPerDay",
                 "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
                 "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
ENGINEERED = ["pulse_pressure", "map_bp"]
FEATURES = BASE_FEATURES + ENGINEERED
SENSITIVE = ["male", "age_group"]

for d in (PROC_DIR, MODEL_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
