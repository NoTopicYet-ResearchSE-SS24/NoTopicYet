from typing import Callable

import pytest
from heartpredict.backend.data import MLData
from heartpredict.backend.ml import MLBackend, load_model


def test_load_pretrained_models_seed_42(ml_data_func: Callable[..., MLData]) -> None:
    data = ml_data_func()
    best_decision_tree_score = 0.984
    decision_tree = load_model(
        f"results/trained_models/DecisionTreeClassifier_model_{data.random_seed}.joblib"
    )
    assert decision_tree.score(data.valid.x, data.valid.y) == best_decision_tree_score

    best_random_forest_score = 0.99
    random_forest = load_model(
        f"results/trained_models/RandomForestClassifier_model_{data.random_seed}.joblib"
    )
    assert random_forest.score(data.valid.x, data.valid.y) == best_random_forest_score

    best_knn_score = 0.977
    knn = load_model(
        f"results/trained_models/KNeighborsClassifier_model_{data.random_seed}.joblib"
    )
    assert knn.score(data.valid.x, data.valid.y) == best_knn_score

    best_lda_score = 0.839
    lda = load_model(
        f"results/trained_models/LinearDiscriminantAnalysis_model_{data.random_seed}.joblib"
    )
    assert lda.score(data.valid.x, data.valid.y) == best_lda_score

    best_qda_score = 0.829
    qda = load_model(
        f"results/trained_models/QuadraticDiscriminantAnalysis_model_{data.random_seed}.joblib"
    )
    assert qda.score(data.valid.x, data.valid.y) == best_qda_score

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model("CoolModel.joblib")
    assert (
            str(exc_info.value) == "[Errno 2] No such file or directory: 'CoolModel.joblib'"
    )


def test_train_model_for_classification_seed_42(
        ml_data_func: Callable[..., MLData],
) -> None:
    data = ml_data_func(random_seed=42)
    backend = MLBackend(data)
    path_to_best_model = backend.classification_for_different_classifiers().model_file

    best_model_accuracy = 0.99
    best_model = load_model(path_to_best_model)
    assert (
            best_model.score(backend.data.valid.x, backend.data.valid.y)
            == best_model_accuracy
    )
