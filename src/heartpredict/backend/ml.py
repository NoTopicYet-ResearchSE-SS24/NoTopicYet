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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class EvaluationMetric:
    name: str
    function: Any
    optimum: Any


@dataclass
class ModelWithParams:
    model: BaseEstimator
    hyperparam_name: str
    values: Any
    model_type: str


@dataclass
class TrainingResult:
    model: BaseEstimator
    model_name: str
    model_classes: Any
    hyperparam_name: str
    best_hyperparam_value: Any


@dataclass
class OptimalModel:
    model: BaseEstimator
    score_name: str
    score: float
    model_file: Path


class MLBackend:
    def __init__(
            self,
            data: MLData,
    ) -> None:
        self.data = data

        self.max_tree_depth = self._calculate_max_tree_depth()
        self.k_min = self._calculate_k_min()

    def classification_for_different_classifiers(self) -> OptimalModel:
        """
        Train different classifiers and return the best performing model.
        Returns:
            OptimalModel: Best performing model of all classifiers.
        """
        classifiers = [
            ModelWithParams(
                DecisionTreeClassifier(random_state=self.data.random_seed),
                "max_depth",
                range(1, self.max_tree_depth),
                "classifier"
            ),
            ModelWithParams(
                RandomForestClassifier(random_state=self.data.random_seed),
                "max_depth",
                range(self.k_min, self.k_min + 10),
                "classifier"
            ),
            ModelWithParams(KNeighborsClassifier(), "n_neighbors", range(3, 7), "classifier"),
            ModelWithParams(LinearDiscriminantAnalysis(), "", None, "classifier"),
            ModelWithParams(QuadraticDiscriminantAnalysis(), "", None, "classifier"),
        ]

        eval_metric = EvaluationMetric("Accuracy", accuracy_score, np.argmax)
        return self._train_models(classifiers, eval_metric)

    def regression_for_different_regressors(self) -> OptimalModel:
        """
        Perform regression with different regressors and return the best performing model.
        Returns:
            OptimalModel: Best performing model of all regressors.
        """
        regressors = [
            ModelWithParams(
                LogisticRegression(random_state=self.data.random_seed),
                "C",
                np.arange(0.2, 2.2, 0.2),
                "regressor"
            ),
            ModelWithParams(
                LogisticRegressionCV(penalty="elasticnet",
                                     solver="saga",
                                     l1_ratios=np.arange(0.1, 1.1, 0.1),
                                     random_state=self.data.random_seed),
                "",
                None,
                "regressor"
            ),
        ]
        eval_metric = EvaluationMetric("Root Mean Squared Error", root_mean_squared_error, np.argmin)
        return self._train_models(regressors, eval_metric)

    def _k_fold_cross_validation(
            self, model: BaseEstimator, hyperparam_name: str, value: Any
    ) -> Any:
        """
        Perform k-fold cross validation.
        Args:
            model: Model to train.
            hyperparam_name: Hyperparameter name.
            value: Value of the hyperparameter.

        Returns:
            Mean accuracy of the model.
        """
        if hyperparam_name:
            model.set_params(**{hyperparam_name: value})
        accuracy = cross_val_score(model, self.data.train.x, self.data.train.y)
        return accuracy.mean()

    def _train_w_best_hyperparam(self, model: ModelWithParams) -> TrainingResult:
        """
        Train a model with the best hyperparameter.
        Args:
            model: Model to train.

        Returns:
            TrainingResult: Best performing model with the best hyperparameter.
        """
        best_hyperparam_value = None
        if model.hyperparam_name and model.values is not None:
            scores = [
                self._k_fold_cross_validation(
                    model.model, model.hyperparam_name, value
                )
                for value in model.values
            ]
            best_hyperparam_value = model.values[np.argmax(scores)]
            model.model.set_params(
                **{model.hyperparam_name: best_hyperparam_value}
            )

        model.model.fit(self.data.train.x, self.data.train.y)
        return TrainingResult(
            model.model,
            type(model.model).__name__,
            model.model.classes_,  # type: ignore
            model.hyperparam_name,
            best_hyperparam_value,
        )

    def _train_model(self, model: ModelWithParams, eval_metric) -> OptimalModel:
        """
        Train a model and return the best performing model.
        Args:
            model: Model to train.
            eval_metric: Evaluation metric to use for model selection.

        Returns:
            OptimalModel: Best performing model for different hyperparameters.
        """
        training_result = self._train_w_best_hyperparam(model)

        y_pred = training_result.model.predict(self.data.valid.x)  # type: ignore
        acc = eval_metric.function(self.data.valid.y, y_pred)
        print(
            f"Best Model for {training_result.model_name}"
            f"with {training_result.hyperparam_name}"
            f"={training_result.best_hyperparam_value}, "
            f"Classes: {training_result.model_classes}: "
            f"{eval_metric.name} Score: {acc}"
        )

        # Save the trained model
        output_dir = Path(f"results/trained_models/{model.model_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_file = (
                output_dir
                / f"{training_result.model_name}_model_{self.data.random_seed}.joblib"
        )
        joblib.dump(training_result.model, model_file, compress=False)
        return OptimalModel(training_result.model, eval_metric.name, float(acc), model_file)

    def _train_models(self, models, eval_metric) -> OptimalModel:
        """
        Train models and return the best performing model.
        Args:
            models: Different models to train.
            eval_metric: Evaluation metric to use for model selection.

        Returns:
            TrainingResult: Best performing model of all models.
        """
        training_results = [self._train_model(m, eval_metric) for m in models]

        scores = [res.score for res in training_results]
        best_performance = eval_metric.optimum(scores)
        print(
            f"Best Model: {type(training_results[best_performance].model).__name__}"
            f"with {eval_metric.name}: "
            f"{training_results[best_performance].score}"
        )
        return training_results[best_performance]

    def _calculate_max_tree_depth(self):
        """
        Calculate the maximum tree depth for decision tree and random forest.
        Early stopping prevents overfitting.

        Returns:
            Maximum tree depth.
        """
        return int(np.log2(self.data.train.x.shape[1])) + 1

    def _calculate_k_min(self):
        """
        Calculate the minimum number of neighbors for KNN.
        Square root of the number of samples is a good starting point for practical applications.

        Returns:
            Minimum number of neighbors.
        """
        return int(np.sqrt(self.data.train.x.shape[0]))


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
    """
    Get the MLBackend instance.
    Args:
        ml_data:

    Returns:
        MLBackend instance.
    """
    return MLBackend(ml_data)
