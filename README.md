# ml_ops_macro_economic

Professional linear regression training pipeline for tabular real-world datasets.

## What this project includes

- Train/validation/test split for honest model assessment.
- Automatic preprocessing:
  - Numeric columns: median imputation + scaling.
  - Categorical columns: most-frequent imputation + one-hot encoding.
- K-fold cross-validation diagnostics on the training split.
- Regression metrics (MAE, RMSE, R²) for validation and test sets.
- Reproducible training with a configurable random seed.
- Saved model artifact (`joblib`) and metrics report (`json`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train a model

```bash
python src/train_linear_regression.py \
  --data-path data/your_dataset.csv \
  --target target_column_name \
  --output-dir artifacts \
  --test-size 0.2 \
  --valid-size 0.25 \
  --cv-folds 5 \
  --random-state 42
```

## Output artifacts

After training:

- `artifacts/linear_regression_pipeline.joblib`: trained sklearn pipeline.
- `artifacts/metrics.json`: cross-validation summary + validation/test metrics.

## Run tests

```bash
pytest -q
```
