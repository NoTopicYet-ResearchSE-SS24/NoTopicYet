from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class NumpyMatrix:
    x: np.ndarray
    y: np.ndarray


class ProjectData:
    def __init__(self,
                 csv: Path = Path("data/heart_failure_clinical_records.csv")) -> None:
        self.df = pd.read_csv(csv)

    @classmethod
    @lru_cache
    def build(
            cls, csv: Path = Path("data/heart_failure_clinical_records.csv")
    ):
        return cls(csv)


class MLData:
    def __init__(
            self, project_data: ProjectData, test_size: float, random_seed: int
    ) -> None:
        self.project_data = project_data
        self.test_size = test_size
        self.random_seed = random_seed
        self.dataset = self._get_whole_dataset()
        self.scaled_feature_matrix = MLData._scale_input_features(self.dataset.x)[0]
        self.train, self.valid = self._get_prepared_matrices()

    @classmethod
    @lru_cache
    def build(
            cls, project_data: ProjectData, test_size: float = 0.2,
            random_seed: int = 42
    ):
        return cls(project_data, test_size, random_seed)

    def _get_whole_dataset(self) -> NumpyMatrix:
        """
        Prepare the whole dataset.
        Returns:
            Whole dataset as NumpyMatrix.
        """
        x = self.project_data.df.drop(columns=["DEATH_EVENT"]).values
        y = self.project_data.df["DEATH_EVENT"].values

        return NumpyMatrix(x, y)  # type: ignore

    def _get_prepared_matrices(self) -> tuple[NumpyMatrix, NumpyMatrix]:
        """
        Prepare training and validation matrices.
        Returns:
            Training and validation matrices as NumpyMatrix.
        """
        unscaled_x_train, unscaled_x_valid, y_train, y_valid = train_test_split(
            self.dataset.x,
            self.dataset.y,
            test_size=self.test_size,
            random_state=self.random_seed,
        )
        x_train, x_valid = MLData._scale_input_features(unscaled_x_train,
                                                        unscaled_x_valid)
        return NumpyMatrix(x_train, y_train), NumpyMatrix(x_valid, y_valid)

    @staticmethod
    def _scale_input_features(
            x_train: np.ndarray, x_valid: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale input features.
        Args:
            x_train:
            x_valid:

        Returns:
            Scaled training and validation input features.
        """
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        # Save the fitted scaler needed for prediction of new data.
        output_dir = Path("results/scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_file = output_dir / "used_scaler.joblib"
        joblib.dump(scaler, scaler_file, compress=False)

        if x_valid is not None:
            x_valid = scaler.transform(x_valid)  # type: ignore
        return x_train, x_valid
