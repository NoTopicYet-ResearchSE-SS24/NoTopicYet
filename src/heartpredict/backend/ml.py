from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from heartpredict.backend.data import MLData
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ClassifierWithParams:
    model: BaseEstimator
    hyperparam_name: str
    values: range | None


@dataclass
class TrainingResult:
    model: BaseEstimator
    model_name: str
    model_classes: Any
    hyperparam_name: str
    best_hyperparam_value: Any


@dataclass
class ClassificationResult:
    model: BaseEstimator
    accuracy: float
    model_file: Path


class MLBackend:
    def __init__(
            self,
            data: MLData,
    ) -> None:
        self.data = data

    def k_fold_cross_validation(
            self, classifier: BaseEstimator, hyperparam_name: str, value: Any
    ) -> Any:
        """
        Perform k-fold cross validation.
        Args:
            classifier: Classifier to train.
            hyperparam_name: Hyperparameter name.
            value: Value of the hyperparameter.

        Returns:
            Any: Mean accuracy score.
        """
        if hyperparam_name:
            classifier.set_params(**{hyperparam_name: value})
        accuracy = cross_val_score(classifier, self.data.train.x, self.data.train.y)
        return accuracy.mean()

    def train_w_best_hyperparam(
            self, classifier: ClassifierWithParams
    ) -> TrainingResult:
        """
        Train the classifier with the best hyperparameter value.
        Args:
            classifier: Classifier to train.

        Returns:
            TrainingResult: Best model with hyperparameter tuning.
        """
        best_hyperparam_value = None
        if classifier.hyperparam_name and classifier.values:
            accuracies = [
                self.k_fold_cross_validation(
                    classifier.model, classifier.hyperparam_name, value
                )
                for value in classifier.values
            ]
            best_hyperparam_value = classifier.values[np.argmax(accuracies)]
            classifier.model.set_params(
                **{classifier.hyperparam_name: best_hyperparam_value}
            )
        classifier.model.fit(self.data.train.x, self.data.train.y)  # type: ignore
        return TrainingResult(
            classifier.model,
            type(classifier.model).__name__,
            classifier.model.classes_,  # type: ignore
            classifier.hyperparam_name,
            best_hyperparam_value,
        )

    def train_classification(
            self, classifier: ClassifierWithParams
    ) -> ClassificationResult:
        """
        Train the classifier and return the best performing model.
        Args:
            classifier: Classifier to train.

        Returns:
            ClassificationResult: Best performing model of hyperparameter tuning.
        """
        training_result = self.train_w_best_hyperparam(classifier)

        y_pred = training_result.model.predict(self.data.valid.x)  # type: ignore
        acc = accuracy_score(self.data.valid.y, y_pred)
        print(
            f"Best Model for {training_result.model_name}"
            f"with {training_result.hyperparam_name}"
            f"={training_result.best_hyperparam_value}, "
            f"Classes: {training_result.model_classes}: Accuracy Score: {acc}"
        )

        # Save the trained model
        output_dir = Path("results/trained_models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_file = (
                output_dir
                / f"{training_result.model_name}_model_{self.data.random_seed}.joblib"
        )
        joblib.dump(training_result.model, model_file, compress=False)
        return ClassificationResult(training_result.model, float(acc), model_file)

    def classification_for_different_classifiers(self) -> ClassificationResult:
        """
        Train different classifiers and return the best performing model.
        Returns:
            ClassificationResult: Best performing model of all classifiers.
        """
        classifiers = [
            ClassifierWithParams(
                DecisionTreeClassifier(random_state=self.data.random_seed),
                "max_depth",
                range(1, 12),
            ),
            ClassifierWithParams(
                RandomForestClassifier(random_state=self.data.random_seed),
                "max_depth",
                range(1, 12),
            ),
            ClassifierWithParams(KNeighborsClassifier(), "n_neighbors", range(3, 7)),
            ClassifierWithParams(LinearDiscriminantAnalysis(), "", None),
            ClassifierWithParams(QuadraticDiscriminantAnalysis(), "", None),
        ]

        # TODO: Use different / combined evaluation metrics
        # TODO: Evaluation based only based on accuracy score leads to overfitting.
        training_results = [self.train_classification(c) for c in classifiers]

        accuracy_scores = [res.accuracy for res in training_results]
        best_performance = np.argmax(accuracy_scores)
        print(
            f"Best Model: {type(training_results[best_performance].model).__name__}"
            "with Accuracy Score: "
            f"{training_results[best_performance].accuracy}"
        )
        return training_results[best_performance]


def load_model(model_file) -> Any:
    """
    Load the trained model.
    Args:
        model_file: Path to the model file.

    Returns:
        Loaded model.
    """
    return joblib.load(model_file)


@lru_cache(typed=True)
def get_ml_backend(ml_data: MLData) -> MLBackend:
    return MLBackend(ml_data)
