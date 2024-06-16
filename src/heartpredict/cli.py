from heartpredict.backend.io import get_ml_matrices
from heartpredict.backend.ml import prepare_train_test_data, classification_for_different_classifiers, load_model

import importlib.metadata

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    print(importlib.metadata.version("heartpredict"))


@app.command()
def test() -> None:
    print("test successful")


@app.command()
def train_model_for_classification(csv: str = "data/heart_failure_clinical_records.csv") -> None:
    x, y = get_ml_matrices(csv)
    x_train, x_test, y_train, y_test = prepare_train_test_data(x, y)
    score, path_to_best_model = classification_for_different_classifiers(x_train, y_train, x_test, y_test)

    # TODO: move this to pytest
    model = load_model(path_to_best_model)
    print(f"Model loaded from {path_to_best_model}")
    print(f"Verify Performance of Best model: {model.score(x_test, y_test)}")
    assert score == model.score(x_test, y_test)
