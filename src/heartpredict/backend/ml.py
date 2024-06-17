import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

RANDOM_SEED = 42


def set_random_seed(seed):
    """
    Set the random seed.
    Args:
        seed:

    Returns:

    """
    global RANDOM_SEED
    RANDOM_SEED = seed


def prepare_train_test_data(x, y):
    """
    Prepare the train and test data.
    Args:
        x:
        y:

    Returns:

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
    x_train, x_test = scale_input_features(x_train, x_test)
    return x_train, x_test, y_train, y_test


def scale_input_features(x_train, x_test):
    """
    Scale the input features.
    Args:
        x_train:
        x_test:

    Returns:

    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def k_fold_cross_validation(classifier, x, y, hyperparam_name, value):
    """
    Perform k-fold cross validation.
    Args:
        classifier:
        x:
        y:
        hyperparam_name:
        value:

    Returns:

    """
    if hyperparam_name:
        classifier.set_params(**{hyperparam_name: value})
    accuracy = cross_val_score(classifier, x, y)
    return accuracy.mean()


def train_w_best_hyperparam(classifier_hyper, x, y):
    """
    Fin the best hyperparameter value and train the classifier.
    Args:
        classifier_hyper:
        x:
        y:

    Returns:

    """
    classifier, hyperparam_name, hyperparam_values = classifier_hyper

    best_hyperparam_value = None
    if hyperparam_name:
        accuracies = [k_fold_cross_validation(classifier, x, y, hyperparam_name, value) for value in hyperparam_values]
        best_hyperparam_value = hyperparam_values[np.argmax(accuracies)]
        classifier.set_params(**{hyperparam_name: best_hyperparam_value})

    classifier.fit(x, y)
    return classifier, hyperparam_name, best_hyperparam_value


def train_classification(classifier_hyper, x_train, y_train, x_test, y_test):
    """
    Train a classifier for classification task with a given hyperparameter.
    Args:
        classifier_hyper:
        x_train:
        y_train:
        x_test:
        y_test:

    Returns:

    """
    model, hyperparam, best_hyperparam_value = train_w_best_hyperparam(classifier_hyper, x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Best Model for {type(model).__name__} with {hyperparam}={best_hyperparam_value}, '
          f'Classes: {model.classes_}: Accuracy Score: {acc}')

    # Save the trained model
    output_dir = Path("results/trained_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / f"{type(model).__name__}_model_{RANDOM_SEED}.joblib"
    joblib.dump(model, model_file, compress=False)
    return model, acc, model_file


def classification_for_different_classifiers(x_train, y_train, x_test, y_test):
    """
    Train different classifiers for classification task.
    Args:
        x_train:
        y_train:
        x_test:
        y_test:

    Returns:
        Path to best performed model and its accuracy score.
    """
    classifiers_hyper = [
        (DecisionTreeClassifier(random_state=RANDOM_SEED), "max_depth", range(1, 12)),
        (RandomForestClassifier(random_state=RANDOM_SEED), "max_depth", range(1, 12)),
        (KNeighborsClassifier(), "n_neighbors", range(3, 7)),
        (LinearDiscriminantAnalysis(), None, None),
        (QuadraticDiscriminantAnalysis(), None, None)
    ]

    # TODO: Use different / combined evaluation metrics
    # TODO: Evaluation based only based on accuracy score leads to overfitting.
    training_results = []  # List of tuples (classifier, accuracy_score, path_to_model)
    for classifier_hyper in classifiers_hyper:
        training_results.append(train_classification(classifier_hyper, x_train, y_train, x_test, y_test))

    accuracy_scores = [score for _, score, _ in training_results]
    best_performance = np.argmax(accuracy_scores)
    print(f'Best Model: {type(classifiers_hyper[best_performance][0]).__name__} with Accuracy Score: '
          f'{training_results[best_performance][1]}')
    return training_results[best_performance][2], training_results[best_performance][1]


def load_model(model_file):
    """
    Load the trained model.
    Args:
        model_file:

    Returns:
        Loaded model.
    """
    return joblib.load(model_file)
