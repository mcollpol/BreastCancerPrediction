"""
Module for making predictions using a saved model pipeline.

This module provides functionality to make predictions using a saved model pipeline.
"""
import typing as t

import pandas as pd

from log_reg_model import __version__ as _version
from log_reg_model.config.core import config
from log_reg_model.processing.data_manager import load_pipeline
from log_reg_model.processing.validation import validate_inputs

# Load the saved model pipeline
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """
    Make a prediction using a saved model pipeline.

    Args:
        input_data (Union[pd.DataFrame, dict]): The input data for making predictions.
            If a DataFrame, it should contain the features required for prediction.
            If a dictionary, it should contain the feature values.

    Returns:
        dict: A dictionary containing the predictions, version information,
        and any errors encountered.
    """
    # Convert input data to a DataFrame.
    data = pd.DataFrame(input_data)

    # Validate input data.
    validated_data, errors = validate_inputs(input_data=data)

    # Initialize results dictionary.
    results = {"predictions": None, "version": _version, "errors": errors}

    # Make predictions if there are no validation errors.
    if not errors:
        predictions = _pipe.predict(X=validated_data[config.model_config.features])

        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results
