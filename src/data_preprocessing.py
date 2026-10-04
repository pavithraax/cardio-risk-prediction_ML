"""Load raw data, clean, stratified split, save to data/processed."""
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (DATA_URL, FEATURES, PROC_DIR, RAW_PATH, SEED, TARGET,
                     TEST_SIZE)
from .features import add_age_group, add_features


def download_if_missing():
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset -> {RAW_PATH}")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)


def load_raw() -> pd.DataFrame:
    download_if_missing()
    return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # Physiologically impossible values -> NaN (imputed later, inside CV)
    df.loc[(df["sysBP"] < df["diaBP"]), ["sysBP", "diaBP"]] = float("nan")
    df.loc[df["BMI"] > 60, "BMI"] = float("nan")
    return df


def get_split():
    """Returns X_train, X_test, y_train, y_test (+ age_group frames for fairness)."""
    df = add_age_group(add_features(clean(load_raw())))
    X, y = df[FEATURES + ["age_group"]], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=SEED)


def save_processed():
    Xtr, Xte, ytr, yte = get_split()
    tr, te = Xtr.assign(TenYearCHD=ytr), Xte.assign(TenYearCHD=yte)
    tr.to_csv(PROC_DIR / "train.csv", index=False)
    te.to_csv(PROC_DIR / "test.csv", index=False)
    print(f"train={tr.shape}, test={te.shape}, "
          f"positive rate train={ytr.mean():.3f} test={yte.mean():.3f}")


if __name__ == "__main__":
    save_processed()
