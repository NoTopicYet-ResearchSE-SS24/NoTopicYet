from functools import lru_cache
import pandas as pd
import pathlib


@lru_cache
def load_data_csv():
    """
    Load the data from the csv file.
    Returns:

    """
    project_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    data = pd.read_csv(project_path / "data/hear_failure_clinical_records_dataset.csv")

    x = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']

    x_matrix = x.values
    y_matrix = y.values
    return x_matrix, y_matrix
