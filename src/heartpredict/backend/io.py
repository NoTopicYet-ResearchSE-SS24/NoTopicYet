from functools import lru_cache
import pandas as pd


@lru_cache
def get_data_frame(csv_file_path: str = "data/heart_failure_clinical_records.csv"):
    """
    Get the data frame.
    Returns:
        Data frame.
    """
    data = pd.read_csv(csv_file_path)
    return data


@lru_cache
def get_ml_matrices(csv_file_path: str = "data/heart_failure_clinical_records.csv"):
    """
    Load the data from the csv file.
    Returns:
        Feature and target matrices.
    """
    df = get_data_frame(csv_file_path)

    x = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    x_matrix = x.values
    y_matrix = y.values
    return x_matrix, y_matrix
