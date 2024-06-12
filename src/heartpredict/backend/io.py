import pandas as pd


def load_data_csv(file_path):
    """
    Load the data from the file.
    Args:
        file_path:

    Returns:

    """
    data = pd.read_csv(file_path)

    x = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']

    x_matrix = x.values
    y_matrix = y.values
    return x_matrix, y_matrix
