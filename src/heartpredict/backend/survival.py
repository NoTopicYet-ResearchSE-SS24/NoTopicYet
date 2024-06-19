from heartpredict.backend.ml import load_model
from heartpredict.backend.io import get_data_frame, get_ml_matrices

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter


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
    scaler = load_model("results/scalers/used_scaler.joblib")
    data = get_data_frame()
    x, y = get_ml_matrices()
    x = scaler.transform(x)

    days_column = data.columns[-2]
    death_event_column = data.columns[-1]

    # Predict probabilities of a death event.
    data['death_event_prob'] = regressor.predict_proba(x)[:, 1]

    # Stratify data based on predicted probabilities.
    data['risk_group'] = pd.qcut(data['death_event_prob'], q=3, labels=["Low Risk", "Medium Risk", "High Risk"])

    # Plot Kaplan-Meier curves for each risk group
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for group, subset in data.groupby('risk_group', observed=True):
        time = subset[days_column]
        event_observed = subset[death_event_column].astype(bool)
        kmf.fit(durations=time, event_observed=event_observed, label=group)
        kmf.plot_survival_function()

    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Survival Curves Stratified by Predicted Risk")
    plt.legend(title="Risk Group")
    plt.grid(True)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'kaplan_meier_plot.png')
    plt.close()
    print(f"Kaplan-Meier plot saved to {output_dir / 'kaplan_meier_plot.png'}")
