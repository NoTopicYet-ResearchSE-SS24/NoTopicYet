from pathlib import Path
from typing import Callable

import pytest
from heartpredict.backend.data import MLData, ProjectData, get_ml_data, get_project_data


@pytest.fixture
def project_data_func() -> Callable[..., ProjectData]:
    def _project_data_factory(
        csv_path: Path = Path("data/heart_failure_clinical_records.csv")
        ) -> ProjectData:
        return get_project_data(csv_path)
    return _project_data_factory


@pytest.fixture
def ml_data_func(project_data_func: Callable[..., ProjectData]) -> Callable[..., MLData]:
    def _ml_data_factory(test_size: float = 0.2, random_seed: int = 42) -> MLData:
        return get_ml_data(project_data_func(), test_size, random_seed)
    return _ml_data_factory

