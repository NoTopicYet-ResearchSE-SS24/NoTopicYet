import pandas as pd
import matplotlib.pyplot as plt

# Read-in dataset

def import_csv(csv_path: str = "data/heart_failure_clinical_records.csv"):
    """Import dataset into Pandas DataFrame"""
    df = pd.read_csv(csv_path)
    return df

def save_variable_distribution(df, col:str):
    """Save unique variable expressions in a dict"""
    distribution = df[col].value_counts().to_dict()
    return distribution


def save_distribution_plot(distribution, variable_name):
    """Create a bar plot and return it in a plot variable"""
    labels = distribution.keys()
    values = distribution.values()  

    # Erstelle das Figure- und Axes-Objekt
    fig, ax = plt.subplots()

    # Erstelle den Bar Plot
    ax.bar(labels, values)
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Count")
    ax.set_title(f"{variable_name} distribution")

    # Speichere den Plot in einer Variable
    plot_variable = (fig, ax)
    return plot_variable


def show_plot(plot_variable):
    """Visualize a given plot variable"""
    fig, ax = plot_variable
    plt.show()


def analyze_subset():
    """
    Present variable distribution for multiple conditions
    E.g. How many patients over age 60 smoked and died after heart-attack 
    """
    pass

df=import_csv()
temp = save_variable_distribution(df,col="age")
plot = save_distribution_plot(temp, variable_name="age")
show_plot(plot)