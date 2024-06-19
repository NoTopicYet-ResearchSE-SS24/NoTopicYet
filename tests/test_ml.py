import pytest

from heartpredict.backend.io import get_ml_matrices
from heartpredict.backend.ml import load_model, prepare_train_valid_data, set_random_seed, \
    classification_for_different_classifiers, regression_for_different_regressors

from sklearn.metrics import root_mean_squared_error

RANDOM_SEED = 42


def test_load_pretrained_classifiers_models_seed_42():
    x, y = get_ml_matrices("data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_valid_data(x, y)

    best_decision_tree_score = 0.859
    decision_tree = load_model(
        f"results/trained_models/classification/DecisionTreeClassifier_model_{RANDOM_SEED}.joblib")
    assert decision_tree.score(x_test, y_test) == best_decision_tree_score

    best_random_forest_score = 0.887
    random_forest = load_model(
        f"results/trained_models/classification/RandomForestClassifier_model_{RANDOM_SEED}.joblib")
    assert random_forest.score(x_test, y_test) == best_random_forest_score

    best_knn_score = 0.821
    knn = load_model(f"results/trained_models/classification/KNeighborsClassifier_model_{RANDOM_SEED}.joblib")
    assert knn.score(x_test, y_test) == best_knn_score

    best_lda_score = 0.839
    lda = load_model(f"results/trained_models/classification/LinearDiscriminantAnalysis_model_{RANDOM_SEED}.joblib")
    assert lda.score(x_test, y_test) == best_lda_score

    best_qda_score = 0.829
    qda = load_model(f"results/trained_models/classification/QuadraticDiscriminantAnalysis_model_{RANDOM_SEED}.joblib")
    assert qda.score(x_test, y_test) == best_qda_score

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model("CoolModel.joblib")
    assert str(exc_info.value) == "[Errno 2] No such file or directory: 'CoolModel.joblib'"


def test_train_model_for_classification_seed_42():
    set_random_seed(RANDOM_SEED)
    x, y = get_ml_matrices("data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_valid_data(x, y)
    path_to_best_model, _ = classification_for_different_classifiers(x_train, y_train, x_test, y_test)

    best_model_accuracy = 0.887
    best_model = load_model(path_to_best_model)
    assert best_model.score(x_test, y_test) == best_model_accuracy


def test_load_pretrained_regression_models_seed_42():
    x, y = get_ml_matrices("data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_valid_data(x, y)

    best_logistic_regression_score = 0.386
    logistic_regressor = load_model(f"results/trained_models/regression/LogisticRegression_model_{RANDOM_SEED}.joblib")
    error = round(root_mean_squared_error(y_test, logistic_regressor.predict(x_test)), 3)
    assert error == best_logistic_regression_score

    best_elastic_net_score = 0.386
    elastic_net = load_model(f"results/trained_models/regression/LogisticRegressionCV_model_{RANDOM_SEED}.joblib")
    error = round(root_mean_squared_error(y_test, elastic_net.predict(x_test)), 3)
    assert error == best_elastic_net_score

def test_train_model_for_regression_seed_42():
    set_random_seed(RANDOM_SEED)
    x, y = get_ml_matrices("data/heart_failure_clinical_records.csv")
    x_train, x_test, y_train, y_test = prepare_train_valid_data(x, y)
    path_to_best_model, _ = regression_for_different_regressors(x_train, y_train, x_test, y_test)

    best_model_rmse = 0.386
    best_model = load_model(path_to_best_model)
    error = round(root_mean_squared_error(y_test, best_model.predict(x_test)), 3)
    assert error == best_model_rmse
