from typing import Callable

from heartpredict.backend.data import MLData


def test_raw_ml_matrices(ml_data_func: Callable[..., MLData]) -> None:
    ml_data = ml_data_func()
    assert ml_data.dataset.x.shape == (5000, 12)
    assert ml_data.dataset.y.shape == (5000,)
