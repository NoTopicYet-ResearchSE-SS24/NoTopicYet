from heartpredict.backend.ml import load_model, scale_input_features
from heartpredict.backend.io import get_data_frame, get_ml_matrices

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sksurv.nonparametric import kaplan_meier_estimator


def create_kaplan_meier_plot(path_to_regressor, out_dir):
    """
    Create a Kaplan-Meier plot stratified by predicted risk groups.
    Args:
        path_to_regressor: Path to the regressor model.
        out_dir: Output directory to save the plot.

    Returns:
        None
    """
    regressor = load_model(path_to_regressor)
    data = get_data_frame()
    x, y = get_ml_matrices()
    x, _ = scale_input_features(x, None)

    days_column = data.columns[-2]
    death_event_column = data.columns[-1]

    # Predict probabilities of a death event.
    data['death_event_prob'] = regressor.predict_proba(x)[:, 1]

    # Stratify data based on predicted probabilities.
    data['risk_group'] = pd.qcut(data['death_event_prob'], q=3, labels=["Low Risk", "Medium Risk", "High Risk"])

    # Plot Kaplan-Meier curves for each risk group
    plt.figure(figsize=(10, 6))
    for group, subset in data.groupby('risk_group', observed=True):
        time = subset[days_column]
        event_observed = subset[death_event_column].astype(bool)
        km_time, km_prob = kaplan_meier_estimator(event_observed, time)
        plt.step(km_time, km_prob, where="post", label=group)

    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.title("Kaplan-Meier Survival Curves Stratified by Predicted Risk")
    plt.legend()
    plt.grid(True)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'kaplan_meier_plot.png')
    print(f"Kaplan-Meier plot saved to {output_dir / 'kaplan_meier_plot.png'}")
