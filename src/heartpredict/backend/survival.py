from heartpredict.backend.data import MLData
from heartpredict.backend.ml import load_model

import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from pathlib import Path
from lifelines import KaplanMeierFitter


class SurvivalBackend:

    def __init__(self, ml_data: MLData) -> None:
        self.df = ml_data.project_data.df
        self.feature_matrix = ml_data.scaled_feature_matrix

    def create_kaplan_meier_plot_for(self, path_to_regressor: Path) -> None:
        """
        Create a Kaplan-Meier plot for specific regressor,
        stratified by predicted risk groups.
        Args:
            path_to_regressor: Path to the saved regressor model.

        Returns:
            None
        """
        regressor = load_model(path_to_regressor)

        days_column = self.df.columns[-2]
        death_event_column = self.df.columns[-1]

        # Predict probabilities of a death event.
        self.df['death_event_prob'] = regressor.predict_proba(
            self.feature_matrix)[:, 1]

        # Stratify data based on predicted probabilities.
        self.df['risk_group'] = pd.qcut(
            self.df['death_event_prob'],
            q=3,
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )

        # Plot Kaplan-Meier curves for each risk group
        plt.figure(figsize=(10, 6))
        kmf = KaplanMeierFitter()

        for group, subset in self.df.groupby('risk_group', observed=True):
            time = subset[days_column]
            event_observed = subset[death_event_column].astype(bool)
            kmf.fit(durations=time, event_observed=event_observed, label=group)
            kmf.plot_survival_function()

        plt.xlabel("Days")
        plt.ylabel("Survival Probability")
        plt.title("Kaplan-Meier Survival Curves Stratified by Predicted Risk")
        plt.legend(title="Risk Group")
        plt.grid(True)

        output_dir = Path("results/survival")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "kaplan_meier_plot.png")
        plt.close()
        print(f"Kaplan-Meier plot saved to "
              f"{output_dir / 'kaplan_meier_plot.png'}")


@lru_cache(typed=True)
def get_survival_backend(ml_data: MLData
                         ) -> SurvivalBackend:
    """
    Get an instance of SurvivalBackend.
    Args:
        ml_data: MLData

    Returns:
        SurvivalBackend instance.
    """
    return SurvivalBackend(ml_data)
