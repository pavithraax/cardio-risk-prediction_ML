"""Gradient-boosted trees (XGBoost) with regularisation + grid search."""
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import CV_FOLDS, FEATURES, MODEL_DIR, SEED
from .data_preprocessing import get_split
from .features import make_preprocessor

PARAM_GRID = {
    "clf__max_depth": [2, 3, 4],
    "clf__learning_rate": [0.025, 0.05, 0.1],
    "clf__n_estimators": [100, 300],
    "clf__reg_lambda": [1, 5],       # L2
    "clf__reg_alpha": [0, 0.5],      # L1
}


def train(cv_folds: int = CV_FOLDS):
    Xtr, Xte, ytr, yte = get_split()
    Xtr = Xtr[FEATURES]
    pipe = Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("clf", XGBClassifier(eval_metric="logloss", subsample=0.8,
                              colsample_bytree=0.8, random_state=SEED,
                              n_jobs=1, tree_method="hist")),
    ])
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    print(f"XGBoost CV AUROC={gs.best_score_:.4f}")
    print("Best params:", gs.best_params_)
    joblib.dump(gs.best_estimator_, MODEL_DIR / "xgboost.pkl")
    return gs.best_estimator_, gs.best_score_, gs.best_params_


if __name__ == "__main__":
    train()
