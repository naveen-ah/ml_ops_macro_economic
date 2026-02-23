import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_linear_regression import train_and_evaluate


class Args:
    def __init__(self, data_path: Path, output_dir: Path):
        self.data_path = data_path
        self.target = "target"
        self.output_dir = output_dir
        self.test_size = 0.2
        self.valid_size = 0.25
        self.random_state = 42
        self.cv_folds = 3


def test_training_pipeline_outputs_artifacts(tmp_path: Path) -> None:
    n = 120
    rng = np.random.default_rng(42)

    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 10, size=n)
    city = np.where(rng.random(size=n) > 0.5, "NY", "SF")
    city_effect = np.where(city == "NY", 4.0, -1.5)

    target = 2.3 * x1 + 0.8 * x2 + city_effect + rng.normal(scale=0.2, size=n)

    data = pd.DataFrame({"x1": x1, "x2": x2, "city": city, "target": target})

    data_path = tmp_path / "train.csv"
    out_dir = tmp_path / "artifacts"
    data.to_csv(data_path, index=False)

    results = train_and_evaluate(Args(data_path=data_path, output_dir=out_dir))

    assert Path(results["model_path"]).exists()
    assert Path(results["metrics_path"]).exists()

    metrics = json.loads(Path(results["metrics_path"]).read_text())
    assert "test_r2" in metrics
    assert metrics["test_r2"] > 0.90
