from heartpredict.backend.io import get_ml_matrices
from heartpredict.backend.survival import create_kaplan_meier_plot
from heartpredict.backend.ml import (prepare_train_valid_data, classification_for_different_classifiers,
                                     set_random_seed, regression_for_different_regressors)

import importlib.metadata

import typer

from typing import Optional

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    print(importlib.metadata.version("heartpredict"))


@app.command()
def test() -> None:
    print("test successful")


@app.command()
def train_model_for_classification(
        seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
        csv: Optional[str] = "data/heart_failure_clinical_records.csv") -> None:
    if seed:
        set_random_seed(seed)
    x, y = get_ml_matrices(csv)
    x_train, x_valid, y_train, y_valid = prepare_train_valid_data(x, y)
    classification_for_different_classifiers(x_train, y_train, x_valid, y_valid)


@app.command()
def train_model_for_regression(
        seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
        csv: Optional[str] = "data/heart_failure_clinical_records.csv") -> None:
    if seed:
        set_random_seed(seed)
    x, y = get_ml_matrices(csv)
    x_train, x_valid, y_train, y_valid = prepare_train_valid_data(x, y)
    regression_for_different_regressors(x_train, y_train, x_valid, y_valid)


@app.command()
def kaplan_meier_plot(seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
                      regressor: Optional[str] = typer.Option(None, "--regressor", help="Path to the regressor model."),
                      csv: Optional[str] = "data/heart_failure_clinical_records.csv",
                      out_dir: Optional[str] = "results/survival") -> None:
    if seed:
        set_random_seed(seed)
    if regressor is None:
        x, y = get_ml_matrices(csv)
        x_train, x_valid, y_train, y_valid = prepare_train_valid_data(x, y)
        regressor, _ = regression_for_different_regressors(x_train, y_train, x_valid, y_valid)
    create_kaplan_meier_plot(regressor, out_dir)
