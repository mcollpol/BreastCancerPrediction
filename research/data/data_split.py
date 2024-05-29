"""
Splits the data in train and test.
Research will be performed only using train.csv.
Test.csv will be used in production for testing purposes.
"""
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


DATASET_FILE = "breast_cancer_wisconsin_diagnostic_dataset.csv"
SCRIPT_DIR = Path(__file__).resolve().parent

def split_and_save_dataset(df: pd.DataFrame,
                           target_column: str,
                           test_size: float,
                           random_state: int) -> None:
    """
    Split the DataFrame into training and testing sets and save them as CSV files.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        None
    """
    # Split the DataFrame into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target_column]),  # predictors
        df[target_column],
        test_size=test_size,
        random_state=random_state,
    )

    # Combine the predictors and target variable for both training and testing sets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save the training and testing sets as CSV files
    train_df.to_csv(SCRIPT_DIR / 'train.csv', index=False)
    test_df.to_csv(SCRIPT_DIR / 'test.csv', index=False)


if __name__ == '__main__':

    data = pd.read_csv(SCRIPT_DIR / DATASET_FILE)

    # Split the dataset and save to CSV files
    split_and_save_dataset(data, 'diagnosis', 0.02 , 42)
