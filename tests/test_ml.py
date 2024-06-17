import pytest

from heartpredict.backend.io import get_ml_matrices
from heartpredict.backend.ml import load_model, prepare_train_test_data, set_random_seed, \
    classification_for_different_classifiers

RANDOM_SEED = 42


def test_load_pretrained_models_seed_42():
    x, y = get_ml_matrices("../data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_test_data(x, y)

    best_decision_tree_score = 0.984
    decision_tree = load_model(f"../results/trained_models/DecisionTreeClassifier_model_{RANDOM_SEED}.joblib")
    assert decision_tree.score(x_test, y_test) == best_decision_tree_score

    best_random_forest_score = 0.99
    random_forest = load_model(f"../results/trained_models/RandomForestClassifier_model_{RANDOM_SEED}.joblib")
    assert random_forest.score(x_test, y_test) == best_random_forest_score

    best_knn_score = 0.977
    knn = load_model(f"../results/trained_models/KNeighborsClassifier_model_{RANDOM_SEED}.joblib")
    assert knn.score(x_test, y_test) == best_knn_score

    best_lda_score = 0.839
    lda = load_model(f"../results/trained_models/LinearDiscriminantAnalysis_model_{RANDOM_SEED}.joblib")
    assert lda.score(x_test, y_test) == best_lda_score

    best_qda_score = 0.829
    qda = load_model(f"../results/trained_models/QuadraticDiscriminantAnalysis_model_{RANDOM_SEED}.joblib")
    assert qda.score(x_test, y_test) == best_qda_score

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model(f"CoolModel.joblib")
    assert str(exc_info.value) == "[Errno 2] No such file or directory: 'CoolModel.joblib'"


def test_train_model_for_classification_seed_42():
    set_random_seed(RANDOM_SEED)
    x, y = get_ml_matrices("../data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_test_data(x, y)
    path_to_best_model, _ = classification_for_different_classifiers(x_train, y_train, x_test, y_test)

    best_model_accuracy = 0.99
    best_model = load_model(path_to_best_model)
    assert best_model.score(x_test, y_test) == best_model_accuracy
