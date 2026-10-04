"""One command to reproduce everything:  python run_all.py"""
from src import data_preprocessing as dp
from src.evaluate import evaluate_all, shap_importance
from src.fairness import fairness_analysis
from src.data_preprocessing import get_split
from src.train_logreg import train as train_lr
from src.train_xgboost import train as train_xgb

if __name__ == "__main__":
    print("== 1. Data ==");           dp.save_processed()
    print("== 2. Baseline LR ==");    _, _, lr_cv = train_lr()
    print("== 3. XGBoost ==");        _, xgb_cv, _ = train_xgb()
    print("== 4. Evaluation ==")
    metrics, probs = evaluate_all({"Logistic Regression (best)": lr_cv, "XGBoost": xgb_cv})
    print(metrics.to_string(index=False))
    print("== 5. SHAP ==");           print(shap_importance().round(4).head(8))
    print("== 6. Fairness ==")
    _, Xte, _, yte = get_split()
    for n, p in probs.items():
        print(n); print(fairness_analysis(Xte, yte, p, n.split()[0]).round(3).to_string(index=False))
