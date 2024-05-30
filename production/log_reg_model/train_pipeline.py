"""
This module defines the training process for the logistic regression model.

It includes functions to:
- Load the dataset.
- Split the dataset into training and testing sets.
- Train the model using a predefined pipeline.
- Save the trained model pipeline.

Example Usage:
    To train the model, simply run this script as follows:
    
    $ python train.py

Functions:
    run_training: Executes the training pipeline and persists the trained model.
"""
from sklearn.model_selection import train_test_split

from log_reg_model.config.core import config
from log_reg_model.pipeline import pipe
from log_reg_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """
    Train the model.

    This function loads the training data, splits it into training and testing sets,
    applies log transformation to the target variable, fits the predefined pipeline
    to the training data, and saves the trained pipeline to a file.

    Raises:
        FileNotFoundError: If the specified data file is not found.
        ValueError: If there are issues with the input data or pipeline configuration.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.train_data_file)

    # divide train and test
    X_train, _ , y_train, _ = train_test_split(
        data.drop(config.columns_to_drop + [config.model_config.target], axis=1),
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.split_seed,
    )

    # fit model
    pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=pipe)

if __name__ == "__main__":
    run_training()
