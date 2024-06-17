import pytest

from heartpredict.backend.io import get_ml_matrices


def test_get_ml_matrices():
    x, y = get_ml_matrices("data/heart_failure_clinical_records.csv")
    assert x.shape == (5000, 12)
    assert y.shape == (5000,)


def test_get_ml_matrices_w_wrong_path():
    with pytest.raises(FileNotFoundError) as exc_info:
        _, _ = get_ml_matrices("MY_WRONG_PATH")

    assert str(exc_info.value) == "[Errno 2] No such file or directory: 'MY_WRONG_PATH'"
