"""Subgroup performance analysis (gender, age group) - mirrors the paper's fairness study."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import FIG_DIR, RAW_PATH, ROOT


def subgroup_report(y, p, groups: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    y, p = np.asarray(y), np.asarray(p)
    for g in groups.dropna().unique():
        mask = (groups == g).values
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        rows.append({"attribute": label, "group": str(g), "n": int(mask.sum()),
                     "incidence": y[mask].mean(),
                     "mean_pred_risk": p[mask].mean(),
                     "auroc": roc_auc_score(y[mask], p[mask]),
                     "auprc": average_precision_score(y[mask], p[mask])})
    return pd.DataFrame(rows)


def fairness_analysis(Xte, yte, p, name="XGBoost") -> pd.DataFrame:
    sex = Xte["male"].map({1: "Male", 0: "Female"})
    rep = pd.concat([subgroup_report(yte, p, sex, "sex"),
                     subgroup_report(yte, p, Xte["age_group"].astype(str), "age_group")])
    rep["calibration_gap"] = rep["mean_pred_risk"] - rep["incidence"]
    rep = rep.sort_values(["attribute", "group"]).reset_index(drop=True)
    rep.round(4).to_csv(ROOT / "results" / f"fairness_{name.lower()}.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lab = rep["attribute"] + ": " + rep["group"]
    ax[0].barh(lab, rep["auroc"], color="steelblue")
    ax[0].set(xlabel="AUROC", title=f"{name}: AUROC by subgroup", xlim=(0.4, 0.9))
    w = 0.38
    idx = np.arange(len(rep))
    ax[1].bar(idx - w / 2, rep["incidence"], w, label="Observed")
    ax[1].bar(idx + w / 2, rep["mean_pred_risk"], w, label="Predicted")
    ax[1].set_xticks(idx)
    ax[1].set_xticklabels(rep["group"], rotation=45)
    ax[1].set(title="Observed vs predicted risk"); ax[1].legend()
    fig.savefig(FIG_DIR / f"fairness_{name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rep
