"""
This module provides utility functions for managing datasets and model pipelines.

Functions:
    load_dataset: Load a dataset from a CSV file and transform the target variable.
    save_pipeline: Persist the trained model pipeline to a file.
    load_pipeline: Load a persisted model pipeline from a file.
    remove_old_pipelines: Remove old model pipelines, keeping only specified files.

Example Usage:
    To load a dataset:
    df = load_dataset(file_name='data.csv')

    To save a pipeline:
    save_pipeline(pipeline_to_persist=trained_pipeline)

    To load a pipeline:
    pipeline = load_pipeline(file_name='pipeline.pkl')

    To remove old pipelines:
    remove_old_pipelines(files_to_keep=['latest_pipeline.pkl'])
"""
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from log_reg_model import __version__ as _version
from log_reg_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and transform the target variable to binary.

    Args:
        file_name (str): The name of the file to load.

    Returns:
        pd.DataFrame: The loaded and transformed DataFrame.
    """
    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    df = df.drop(columns=config.app_config.columns_to_drop)
    # Transforming the target variable to binary.
    df[config.model_config.target] = (df[config.model_config.target]
                                      .map(config.model_config.target_map))
    return df


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """
    Persist the pipeline.

    Saves the versioned model and overwrites any previous saved models.
    This ensures that when the package is published, there is only one
    trained model that can be called, and we know exactly how it was built.

    Args:
        pipeline_to_persist (Pipeline): The pipeline to save.

    Returns:
        None
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """
    Load a persisted pipeline.

    Args:
        file_name (str): The name of the file to load.

    Returns:
        Pipeline: The loaded pipeline.
    """
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one mapping between the package
    version and the model version to be imported and used by other applications.

    Args:
        files_to_keep (t.List[str]): List of files to keep.

    Returns:
        None
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
