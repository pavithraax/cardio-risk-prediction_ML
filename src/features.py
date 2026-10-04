"""Feature engineering and preprocessing pipelines."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful derived features (no target leakage)."""
    df = df.copy()
    df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
    df["map_bp"] = df["diaBP"] + (df["sysBP"] - df["diaBP"]) / 3.0
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Age bands used only for subgroup/fairness analysis."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[0, 44, 54, 64, 120],
                             labels=["<45", "45-54", "55-64", "65+"])
    return df


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Median-impute (fit on train only); optionally standardise (for LR)."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), FEATURES)])
