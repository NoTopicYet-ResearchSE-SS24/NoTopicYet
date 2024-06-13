from functools import lru_cache
import pandas as pd
import pathlib


@lru_cache
def get_data_frame():
    """
    Get the data frame.
    Returns:

    """
    project_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    data = pd.read_csv(project_path / "data/hear_failure_clinical_records_dataset.csv")
    return data


@lru_cache
def get_ml_matrices():
    """
    Load the data from the csv file.
    Returns:

    """
    df = get_data_frame()

    x = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    x_matrix = x.values
    y_matrix = y.values
    return x_matrix, y_matrix
