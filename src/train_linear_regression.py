from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a production-ready linear regression model on tabular data."
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store model and metrics artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for final test set",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.25,
        help="Fraction of train+valid split reserved for validation",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="K-folds on train split for stability diagnostics",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", num_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                ),
            ]
        )
        transformers.append(("categorical", cat_pipe, categorical_features))

    if not transformers:
        raise ValueError("No trainable features found in dataset.")

    return ColumnTransformer(transformers=transformers)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, float]]:
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        preprocessor = build_preprocessor(X_t)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(X_t, y_t)
        y_hat = model.predict(X_v)

        metrics = regression_metrics(y_v, y_hat)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    return fold_metrics


def summarize_cv(cv_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["mae", "rmse", "r2"]
    return {
        f"cv_{k}_mean": float(np.mean([m[k] for m in cv_metrics])) for k in keys
    } | {f"cv_{k}_std": float(np.std([m[k] for m in cv_metrics])) for k in keys}


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not present in dataset columns.")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=args.valid_size,
        random_state=args.random_state,
    )

    cv_metrics = run_cv(X_train, y_train, args.cv_folds, args.random_state)
    cv_summary = summarize_cv(cv_metrics)

    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)

    valid_metrics = {f"valid_{k}": v for k, v in regression_metrics(y_valid, valid_pred).items()}
    test_metrics = {f"test_{k}": v for k, v in regression_metrics(y_test, test_pred).items()}

    metrics = {
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "target": args.target,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "valid_size": args.valid_size,
        "cv_folds": args.cv_folds,
        **cv_summary,
        **valid_metrics,
        **test_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "linear_regression_pipeline.joblib"
    metrics_path = args.output_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    LOGGER.info("Saved model: %s", model_path)
    LOGGER.info("Saved metrics: %s", metrics_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main() -> None:
    configure_logging()
    args = parse_args()

    results = train_and_evaluate(args)
    LOGGER.info("Validation R²: %.4f", results["metrics"]["valid_r2"])
    LOGGER.info("Test R²: %.4f", results["metrics"]["test_r2"])


if __name__ == "__main__":
    main()
