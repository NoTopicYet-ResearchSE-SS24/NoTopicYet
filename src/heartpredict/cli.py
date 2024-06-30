from heartpredict.backend.ml import MLBackend
from heartpredict.backend.data import MLData, ProjectData
from heartpredict.backend.survival import SurvivalBackend

import importlib.metadata

import typer

from pathlib import Path
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
        seed: Optional[int] = typer.Option(42, "--seed",
                                           help="Random seed for reproducibility."),
        csv: Optional[str] = "data/heart_failure_clinical_records.csv") -> None:
    project_data = ProjectData(Path(csv))
    data = MLData(project_data, 0.2, seed)
    backend = MLBackend(data)
    backend.classification_for_different_classifiers()


@app.command()
def train_model_for_regression(
        seed: Optional[int] = typer.Option(42, "--seed",
                                           help="Random seed for reproducibility."),
        csv: Optional[str] = "data/heart_failure_clinical_records.csv") -> None:
    project_data = ProjectData(Path(csv))
    data = MLData(project_data, 0.2, seed)
    backend = MLBackend(data)
    backend.regression_for_different_regressors()


@app.command()
def create_kaplan_meier_plot(
        seed: Optional[int] = typer.Option(42, "--seed",
                                           help="Random seed for reproducibility."),
        regressor: Optional[str] = typer.Option(None, "--regressor",
                                                help="Path to the regressor model."),
        csv: Optional[str] = "data/heart_failure_clinical_records.csv",
) -> None:
    project_data = ProjectData(Path(csv))
    ml_data = MLData(project_data, 0.2, seed)
    survival_backend = SurvivalBackend(ml_data)
    if regressor is None:
        ml_backend = MLBackend(ml_data)
        regressor = ml_backend.regression_for_different_regressors().model_file
    survival_backend.create_kaplan_meier_plot_for(regressor)
