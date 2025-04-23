# In this task, you will do some exploratory data analysis (EDA). Explore the dataset, count,
# summarize, plot things, and report findings that are useful for creating predictions. Remem-
# ber that EDA is not necessarily done once at the start of the project. It is expected that you do
# some EDA, build some features, train some models, then some idea comes up, do some more
# EDA, modify your features, train another model on these new features, and so on

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class  EDA:
    def main():
        # Load the dataset
        df = pd.read_csv('data.csv')

        # Display the first few rows of the dataset
        print(df.head())

        # Display the shape of the dataset
        print(f'Shape of the dataset: {df.shape}')

        # Display the data types of each column
        print(df.dtypes)

        # Display summary statistics of the dataset
        print(df.describe())

        # Check for missing values
        print(df.isnull().sum())

        # Visualize the distribution of numerical features
        df.hist(figsize=(12, 10))
        plt.tight_layout()
        plt.show()