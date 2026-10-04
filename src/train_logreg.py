"""Baseline: regularised logistic regression (Ridge / Lasso / ElasticNet)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, FEATURES, MODEL_DIR, SEED
from .data_preprocessing import get_split
from .features import make_preprocessor


def build(penalty_cfg):
    clf = LogisticRegression(solver="saga", max_iter=5000, random_state=SEED,
                             **penalty_cfg)
    return Pipeline([("prep", make_preprocessor(scale=True)), ("clf", clf)])


def train():
    Xtr, Xte, ytr, yte = get_split()
    Xtr, Xte = Xtr[FEATURES], Xte[FEATURES]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    Cs = np.logspace(-3, 1, 9)
    variants = {
        "Ridge":      {"l1_ratio": 0.0},
        "Lasso":      {"l1_ratio": 1.0},
        "ElasticNet": {"l1_ratio": 0.5},
    }
    results, best_name, best_gs = {}, None, None
    for name, cfg in variants.items():
        gs = GridSearchCV(build(cfg), {"clf__C": Cs}, scoring="roc_auc",
                          cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        results[name] = gs.best_score_
        print(f"{name:11s} CV AUROC={gs.best_score_:.4f} "
              f"(C={gs.best_params_['clf__C']:.4g})")
        if best_gs is None or gs.best_score_ > best_gs.best_score_:
            best_name, best_gs = name, gs
    print(f"Best baseline: {best_name}")
    joblib.dump(best_gs.best_estimator_, MODEL_DIR / "logreg.pkl")
    return best_gs.best_estimator_, results, best_gs.best_score_


if __name__ == "__main__":
    train()
