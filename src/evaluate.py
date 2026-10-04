"""Metrics, ROC/PR curves, confusion matrices, SHAP importance."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             ConfusionMatrixDisplay, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import CV_FOLDS, FEATURES, FIG_DIR, METRICS_PATH, MODEL_DIR, SEED
from .data_preprocessing import get_split


def choose_threshold(model, Xtr, ytr):
    """Youden-J threshold from out-of-fold TRAIN predictions (no test leakage)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(ytr, p)
    return float(thr[np.argmax(tpr - fpr)])


def bootstrap_auc_ci(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def evaluate_all(cv_scores=None):
    Xtr, Xte, ytr, yte = get_split()
    Xtr_f, Xte_f = Xtr[FEATURES], Xte[FEATURES]
    models = {"Logistic Regression (best)": joblib.load(MODEL_DIR / "logreg.pkl"),
              "XGBoost": joblib.load(MODEL_DIR / "xgboost.pkl")}
    rows, probs = [], {}
    fig_roc, ax_roc = plt.subplots(figsize=(5.5, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(5.5, 5))
    for name, m in models.items():
        p = m.predict_proba(Xte_f)[:, 1]
        probs[name] = p
        thr = choose_threshold(m, Xtr_f, ytr)
        pred = (p >= thr).astype(int)
        lo, hi = bootstrap_auc_ci(yte, p)
        rows.append({
            "model": name,
            "cv_auroc": (cv_scores or {}).get(name, np.nan),
            "test_auroc": roc_auc_score(yte, p),
            "auroc_95ci": f"[{lo:.3f}, {hi:.3f}]",
            "test_auprc": average_precision_score(yte, p),
            "brier": brier_score_loss(yte, p),
            "threshold": thr,
            "f1": f1_score(yte, pred),
        })
        fpr, tpr, _ = roc_curve(yte, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={rows[-1]['test_auroc']:.3f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        ax_pr.plot(rc, pr, label=f"{name} (AP={rows[-1]['test_auprc']:.3f})")
        ConfusionMatrixDisplay(confusion_matrix(yte, pred)).plot(colorbar=False)
        plt.title(f"{name}\nthreshold={thr:.3f}")
        plt.savefig(FIG_DIR / f"confusion_{name.split()[0].lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC curves (test set)")
    ax_roc.legend(loc="lower right")
    fig_roc.savefig(FIG_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    ax_pr.axhline(yte.mean(), color="k", ls="--", lw=0.8)
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall (test set)")
    ax_pr.legend()
    fig_pr.savefig(FIG_DIR / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    df = pd.DataFrame(rows).round(4)
    df.to_csv(METRICS_PATH, index=False)
    return df, probs


def shap_importance():
    """SHAP summary + bar plot for the XGBoost model."""
    import shap
    Xtr, Xte, _, _ = get_split()
    m = joblib.load(MODEL_DIR / "xgboost.pkl")
    prep, clf = m.named_steps["prep"], m.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(Xte[FEATURES]), columns=FEATURES)
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    for kind, fname in [("dot", "shap_summary.png"), ("bar", "shap_bar.png")]:
        plt.figure()
        shap.summary_plot(sv, Xt, plot_type=kind, show=False)
        plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    return imp
