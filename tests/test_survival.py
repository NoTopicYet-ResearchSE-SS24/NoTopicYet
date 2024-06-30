from heartpredict.backend.survival import SurvivalBackend
from heartpredict.backend.data import MLData

from typing import Callable
from pathlib import Path

RANDOM_SEED = 42


def test_use_pretrained_regression_model_and_create_kaplan_meier_plot_42(
        ml_data_func: Callable[..., MLData]) -> None:
    regressor_dir = Path("results/trained_models/regressor/"
                         "LogisticRegression_model_42.joblib")
    ml_data = ml_data_func()
    survival_backend = SurvivalBackend(ml_data)
    survival_backend.create_kaplan_meier_plot_for(regressor_dir)
    assert Path("results/survival/kaplan_meier_plot.png").exists()
