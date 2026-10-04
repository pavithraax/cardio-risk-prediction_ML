"""Load the trained model and score a single patient."""
import joblib
import pandas as pd

from .config import FEATURES, MODEL_DIR
from .features import add_features


def load_model(name: str = "xgboost"):
    return joblib.load(MODEL_DIR / f"{name}.pkl")


def predict_risk(patient: dict, model=None) -> float:
    """patient: dict with the 15 base fields. Returns 10-year CHD probability."""
    model = model or load_model()
    df = add_features(pd.DataFrame([patient]))
    return float(model.predict_proba(df[FEATURES])[0, 1])


if __name__ == "__main__":
    demo = dict(male=1, age=58, education=2, currentSmoker=1, cigsPerDay=20,
                BPMeds=0, prevalentStroke=0, prevalentHyp=1, diabetes=0,
                totChol=240, sysBP=150, diaBP=95, BMI=28.5, heartRate=78,
                glucose=90)
    print(f"Predicted 10-year CHD risk: {predict_risk(demo):.1%}")
