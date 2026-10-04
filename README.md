# Predicting Cardiovascular Risk using Health Records (UE24CS352A ML Mini-Project)

Predicts 10-year coronary heart disease (CHD) risk from the **Framingham Heart Study**
dataset (4,240 patients, 15 clinical features, ~15% positive class), following the
methodology of Nguyen (CS229, 2019): regularised **Logistic Regression** baseline vs
**XGBoost**, 80/20 split, 10-fold CV, AUROC as primary metric, plus a **subgroup fairness** analysis.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The dataset is already in `data/raw/framingham.csv` (auto-downloaded if missing).

## Run
```bash
python run_all.py                  # full pipeline: data -> train -> evaluate -> SHAP -> fairness
streamlit run app/demo.py          # interactive demo
python -m src.predict              # CLI prediction for a sample patient
jupyter notebook notebooks/        # 01_eda, 02_baseline, 03_xgboost_tuning
```
Outputs: `models/*.pkl`, `results/metrics.csv`, `results/fairness_*.csv`, `results/figures/*.png`.

## Structure
```
data/raw, data/processed   dataset and train/test CSVs
notebooks/                 EDA, baseline, XGBoost + evaluation + fairness
src/                       config, preprocessing, features, training, evaluation, fairness, predict
app/demo.py                Streamlit demo
models/ results/           saved models, metrics, figures
run_all.py                 end-to-end reproducible pipeline
```
## Notes
- Imputation/scaling happen inside sklearn Pipelines (fit on training folds only) -> no leakage.
- Decision threshold is chosen from out-of-fold *training* predictions, not the test set.
- Team members: Pavithra Reddy / PES1UG24CS489
                Vibhav M      /  PES1UG24CS527
