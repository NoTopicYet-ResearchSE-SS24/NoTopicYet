from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Self

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
    def __init__(self, csv: Path) -> None:
        self.project_path = Path(__file__).parent.parent.parent.parent
        self.df = pd.read_csv(self.project_path.joinpath(csv))


    @classmethod    
    @lru_cache
    def build(
        cls, csv: Path = Path("data/heart_failure_clinical_records.csv")
        ) -> Self:
        return cls(csv)


class MLData:
    def __init__(
        self, project_data: ProjectData, test_size: float, random_seed: int
    ) -> None:
        self.project_data = project_data
        self.test_size = test_size
        self.random_seed = random_seed
        self.raw = self._get_raw_matrix(project_data.df)
        self.train, self.test = self._get_prepared_matrices()

    @classmethod
    @lru_cache
    def build(
        cls, project_data: ProjectData, test_size: float = 0.2, random_seed: int = 42
    ) -> Self:
        return cls(project_data, test_size, random_seed)

    def _get_raw_matrix(self, df: pd.DataFrame) -> NumpyMatrix:
        x = df.drop(columns=["DEATH_EVENT"])
        y = df["DEATH_EVENT"]

        return NumpyMatrix(x.values, y.values)  # type: ignore

    def _get_prepared_matrices(self) -> tuple[NumpyMatrix, NumpyMatrix]:
        unscaled_x_train, unscaled_x_test, y_train, y_test = train_test_split(
            self.raw.x,
            self.raw.y,
            test_size=self.test_size,
            random_state=self.random_seed,
        )
        x_train, x_test = self._scale_input_features(unscaled_x_test, unscaled_x_train)
        return NumpyMatrix(x_train, y_train), NumpyMatrix(x_test, y_test)

    def _scale_input_features(
        self, x_train: np.ndarray, x_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale the input features.
        Args:
            x_train:
            x_test:

        Returns:

        """
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        # Save the fitted scaler needed for prediction of new data.
        output_dir = Path("results/scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_file = output_dir / "used_scaler.joblib"
        joblib.dump(scaler, scaler_file, compress=False)

        x_test = scaler.transform(x_test)  # type: ignore
        return x_train, x_test
