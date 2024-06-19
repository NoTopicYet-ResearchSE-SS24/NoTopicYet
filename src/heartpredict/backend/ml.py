import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, train_test_split

RANDOM_SEED = 42


def set_random_seed(seed):
    """
    Set the random seed.
    Args:
        seed: Random seed.

    Returns:

    """
    global RANDOM_SEED
    RANDOM_SEED = seed


def calculate_max_tree_depth(n_features):
    """
    Calculate the maximum tree depth for decision tree and random forest.
    Early stopping prevents overfitting.
    Args:
        n_features: Number of features.

    Returns:
        Maximum tree depth.
    """
    return int(np.log2(n_features)) + 1


def calculate_k_min(n_samples):
    """
    Calculate the minimum number of neighbors for KNN.
    Square root of the number of samples is a good starting point for practical applications.
    Args:
        n_samples: Number of samples.

    Returns:
        Minimum number of neighbors.
    """
    return int(np.sqrt(n_samples))


def load_model(model_file):
    """
    Load the trained model.
    Args:
        model_file: Path to the model file.

    Returns:
        Loaded model.
    """
    return joblib.load(model_file)


def prepare_train_valid_data(x, y):
    """
    Prepare the train and test data.
    Args:
        x: Feature data.
        y: Target data.

    Returns:
        x_train, x_valid, y_train, y_valid
    """
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
    x_train, x_valid = scale_input_features(x_train, x_valid)
    return x_train, x_valid, y_train, y_valid


def scale_input_features(x_train, x_valid):
    """
    Scale the input features.
    Args:
        x_train: Training feature data.
        x_valid: Validation feature data.

    Returns:
        Scaled training and validation feature data.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    # Save the fitted scaler needed for prediction of new data.
    output_dir = Path("results/scalers")
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_file = output_dir / "used_scaler.joblib"
    joblib.dump(scaler, scaler_file, compress=False)

    x_valid = scaler.transform(x_valid)
    return x_train, x_valid


def k_fold_cross_validation(model, x, y, hyperparam_name, value):
    """
    Perform k-fold cross validation on given feature and target data.
    Args:
        model: ML model.
        x: Feature data.
        y: Target data.
        hyperparam_name: Hyperparameter name for the specific model.
        value: Hyperparameter value.

    Returns:
        Mean accuracy score.
    """
    if hyperparam_name:
        model.set_params(**{hyperparam_name: value})
    accuracy = cross_val_score(model, x, y)
    return accuracy.mean()


def train_w_best_hyperparam(model_hyper, x, y):
    """
    Fin the best hyperparameter value and train the model.
    Args:
        model_hyper: Tuple of model, hyperparameter name and hyperparameter values.
        x: Feature data.
        y: Target data.

    Returns:
        Trained model, hyperparameter name and best hyperparameter value.
    """
    model, hyperparam_name, hyperparam_values = model_hyper

    best_hyperparam_value = None
    if hyperparam_name:
        accuracies = [k_fold_cross_validation(model, x, y, hyperparam_name, value) for value in hyperparam_values]
        best_hyperparam_value = hyperparam_values[np.argmax(accuracies)]
        model.set_params(**{hyperparam_name: best_hyperparam_value})

    model.fit(x, y)
    return model, hyperparam_name, best_hyperparam_value


def train_model(model_hyper, x_train, y_train, x_valid, y_valid, prediction_task):
    """
    Train a classifier for classification task with a given hyperparameter.
    Args:
        model_hyper: Tuple of model, hyperparameter name and hyperparameter values.
        x_train: Training feature data.
        y_train: Training target data.
        x_valid: Validation feature data.
        y_valid: Validation target data.
        prediction_task: Dictionary that contains the name of the task, evaluation metric name and evaluation metric.

    Returns:
        Trained model, score and path to the model file.
    """
    model, hyperparam, best_hyperparam_value = train_w_best_hyperparam(model_hyper, x_train, y_train)

    # Evaluate the model for never seen data.
    y_pred = model.predict(x_valid)
    score_on_validation = prediction_task['eval_metric'](y_valid, y_pred)
    eval_metric_name = prediction_task['eval_metric_name']
    print(f'Best Model for {type(model).__name__} with {hyperparam}={best_hyperparam_value}, {eval_metric_name} for '
          f'Validation Set: {score_on_validation}')

    # Save the trained model
    prediction_task_name = prediction_task['name']
    output_dir = Path(f"results/trained_models/{prediction_task_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / f"{type(model).__name__}_model_{RANDOM_SEED}.joblib"
    joblib.dump(model, model_file, compress=False)
    return model, score_on_validation, model_file


def train_models(models_hyper, x_train, y_train, x_valid, y_valid, prediction_task):
    """
    Train multiple ML models.
    Args:
        models_hyper: List of tuples of model, hyperparameter name and hyperparameter values.
        x_train: Feature for training.
        y_train: Target for training.
        x_valid: Feature for validation.
        y_valid: Target for validation.
        prediction_task: Dictionary that contains the name of the task, evaluation metric name and evaluation metric.

    Returns:

    """
    training_results = []  # List of tuples (regressor, score, path_to_model)
    for regressor_hyper in models_hyper:
        training_results.append(
            train_model(regressor_hyper, x_train, y_train, x_valid, y_valid, prediction_task))

    eval_metric_name = prediction_task['eval_metric_name']
    scores = [score for _, score, _ in training_results]
    best_performance = prediction_task['evaluation'](scores)
    print(f'Best Model: {type(models_hyper[best_performance][0]).__name__} with {eval_metric_name} on Validation Set: '
          f'{training_results[best_performance][1]}')
    return training_results[best_performance][2], training_results[best_performance][1]


def classification_for_different_classifiers(x_train, y_train, x_valid, y_valid):
    """
    Train different classifiers for classification task.
    Args:
        x_train: Training feature data.
        y_train: Training target data.
        x_valid: Validation feature data.
        y_valid: Validation target data.

    Returns:
        Path to best performed model and its accuracy score.
    """
    classification_task = {
        "name": "classification",
        "eval_metric_name": "accuracy",
        "eval_metric": accuracy_score,
        "evaluation": np.argmax
    }

    max_tree_depth = calculate_max_tree_depth(x_train.shape[1])
    k_min = calculate_k_min(x_train.shape[0])

    classifiers_hyper = [
        (DecisionTreeClassifier(random_state=RANDOM_SEED), "max_depth", range(1, max_tree_depth)),
        (RandomForestClassifier(random_state=RANDOM_SEED), "max_depth", range(1, max_tree_depth)),
        (KNeighborsClassifier(), "n_neighbors", range(k_min, k_min + 10)),
        (LinearDiscriminantAnalysis(), None, None),
        (QuadraticDiscriminantAnalysis(), None, None)
    ]

    return train_models(classifiers_hyper, x_train, y_train, x_valid, y_valid, classification_task)


def regression_for_different_regressors(x_train, y_train, x_valid, y_valid):
    """
    Train different regressors for regression task.
    Args:
        x_train: Training feature data.
        y_train: Training target data.
        x_valid: Validation feature data.
        y_valid: Validation target data.

    Returns:
        Path to best performed model and its rmse score.
    """
    regression_task = {
        "name": "regression",
        "eval_metric_name": "rmse",
        "eval_metric": root_mean_squared_error,
        "evaluation": np.argmin
    }

    regressors_hyper = [
        (LogisticRegression(random_state=RANDOM_SEED), "C", np.arange(0.2, 2.2, 0.2)),
        (LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=np.arange(0.1, 1.1, 0.1),
                              random_state=RANDOM_SEED), None, None),
    ]

    return train_models(regressors_hyper, x_train, y_train, x_valid, y_valid, regression_task)
