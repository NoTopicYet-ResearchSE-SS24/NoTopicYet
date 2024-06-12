import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split


def prepare_train_test_data(x, y):
    """
    Prepare the train and test data.
    Args:
        x:
        y:

    Returns:

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def scale_input_features(scaler=StandardScaler(), x_train, x_test):
    """
    Scale the input features.
    Args:
        scaler:
        x_train:
        x_test:

    Returns:

    """
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
    return accuracy


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

    accuracies = Parallel(n_jobs=-1)(
        delayed(classifier)(classifier, x, y, hyperparam_name, value) for value in hyperparam_values
    )

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
    x_train, x_test = scale_input_features(StandardScaler(), x_train, x_test)

    classifier, hyperparam, hyperparam_value = train_w_best_hyperparam(classifier_hyper, x_train, y_train, x_test,
                                                                       y_test)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Best Model for {type(classifier).__name__} with {hyperparam}={hyperparam_value}, '
          f'Classes: {classifier.classes_}: Accuracy Score: {acc}')
    return classifier, acc


def classification_for_different_classifiers(x_train, y_train, x_test, y_test):
    """
    Train different classifiers for classification task.
    Args:
        x_train:
        y_train:
        x_test:
        y_test:

    Returns:
        Best performed model.
    """
    classifiers_hyper = [
        (DecisionTreeClassifier(), "max_depth", range(1, 12)),
        (RandomForestClassifier(), "max_depth", range(1, 12)),
        (KNeighborsClassifier(), "n_neighbors", range(3, 7)),
        (LinearDiscriminantAnalysis(), None, None),
        (QuadraticDiscriminantAnalysis(), None, None)
    ]

    # TODO: Use different / combined evaluation metrics
    models_scores = []
    for classifier_hyper in classifiers_hyper:
        models_scores.append(train_classification(classifier_hyper, x_train, y_train, x_test, y_test))

    accuracy_scores = [score for _, score in models_scores]
    best_performance = np.argmax(accuracy_scores)
    print(f'Best Model: {type(classifiers_hyper[best_performance][0]).__name__} with Accuracy Score: '
          f'{models_scores[best_performance][1]}')
    return models_scores[best_performance][0]
