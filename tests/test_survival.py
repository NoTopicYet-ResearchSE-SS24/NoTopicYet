from heartpredict.backend.survival import create_kaplan_meier_plot

from pathlib import Path

RANDOM_SEED = 42


def test_use_pretrained_regression_model_and_create_kaplan_meier_plot_42():
    create_kaplan_meier_plot(
        f"results/trained_models/regression/LogisticRegression_model_{RANDOM_SEED}.joblib",
        "results/survival"
    )
    assert Path("results/survival/kaplan_meier_plot.png").exists()
