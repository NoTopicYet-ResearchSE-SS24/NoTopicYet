"""Utilities for conducting a descriptive data analysis"""

import matplotlib.pyplot as plt

MEANING_BINARY_COLUMNS = {
    "anaemia": {0: "No anaemia", 1: "anaemia"},
    "diabetes": {0: "No diabetes", 1: "diabetes"},
    "high_blood_pressure": {0: "Normal blood pressure", 1: "High blood pressure"},
    "sex": {0: "Female", 1: "Male"},
    "smoking": {0: "Not smoking", 1: "Is smoking"},
    "DEATH_EVENT": {0: "Survived", 1: "Died"}
}


def calculate_basic_statistics(df, col:str):
    """
    Determine basic statistics for a DataFrame column

    Args:
        df: DataFrame of (sub)dataset
        col: Column to be analyzed

    Returns:
        Dictionary with statistical infos
    """
    # Save and return the results as a dict
    results = {}

    # Check if feature is binary
    unique_values = df[col].unique()
    is_binary = all(value in [0, 1] for value in unique_values)

    if is_binary:
        size = len(df)
        distribution = save_variable_distribution(df=df, col=col)

        # Access mapping for binary digits meaning
        meaning_zero = MEANING_BINARY_COLUMNS[col][0]
        meaning_one = MEANING_BINARY_COLUMNS[col][1]

        results["Feature name"] = col
        results[meaning_zero] = distribution[0] / size
        results[meaning_one] = distribution[1] / size

    elif not is_binary:
        attributes = df[col].describe()

        results["Feature name"] = col
        results["Minimum"] = attributes["min"]
        results["Maximum"] = attributes["max"]
        results["Mean"] = attributes["mean"]
        results["Median"] = attributes["50%"]
        results["Standard deviation"] = attributes["std"]
        
    return results


def save_variable_distribution(df, col:str):
    """
    Save unique variable expressions of a DataFrame column in a dict
    
    Args:
        df: DataFrame of (sub)dataset
        col: Column to be analyzed

    Returns:
        Dictionary counting the variable expressions
    """
    distribution = df[col].value_counts().to_dict()
    return distribution


def save_distribution_plot(distribution:dict, col_name:str):
    """
    Create and return a simple bar plot for a specific column
    
    Args:
        distribution: Dictionary counting the variable expressions
        col_name: Column name for plot description

    Returns:
        Tuple for the plot variables (fig, ax)
    """
    labels = distribution.keys()
    values = distribution.values()  

    # Create a figure and ax object
    fig, ax = plt.subplots()

    # Create a Bar Plot
    ax.bar(labels, values)
    ax.set_xlabel(col_name)
    ax.set_ylabel("Count")
    ax.set_title(f"{col_name} distribution")

    # Save the Bar Plot to a variable
    plot_variable = (fig, ax)
    return plot_variable


def show_plot(plot_variable:tuple):
    """
    Visualize a given plot variable

    
    Args:
        plot_variable: Tuple for the plot variables (fig, ax)

    Returns:
        None
        Prints plot
    """
    fig, ax = plot_variable
    plt.show()


def create_conditional_dataset(df, col:str, num:int, rel:str):
    """
    Create a filtered dataset, e.g. only patients over 60

    Args:
        df: DataFrame of (sub)dataset
        col: Column name for filtering (e.g. 'age')
        num: Number for filtering (e.g. 60)
        rel: Relation for filtering (e.g. ">")

    Returns:
        DataFrame of filtered dataset
    """

    # Check the condition's relation
    if rel == "==":
        cond = df[col] == num
    elif rel == "<":
        cond = df[col] < num
    elif rel == ">":
        cond = df[col] > num
    elif rel == "<=":
        cond = df[col] <= num
    elif rel == ">=":
        cond = df[col] >= num

    # Return conditioned dataset
    df_cond = df[cond].copy()
    return df_cond
