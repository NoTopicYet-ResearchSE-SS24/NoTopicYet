from heartpredict.backend.io import get_ml_matrices
from heartpredict.backend.ml import (prepare_train_test_data, classification_for_different_classifiers,
                                     set_random_seed, scale_input_features)

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
    x_train, x_test, y_train, y_test = prepare_train_test_data(x, y)
    classification_for_different_classifiers(x_train, y_train, x_test, y_test)
