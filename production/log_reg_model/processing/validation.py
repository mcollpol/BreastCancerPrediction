"""
Module for data validation.

This module contains functions and classes for validating input data
for model training and prediction.

Functions:
    drop_na_inputs: Check model inputs for NA values and filter.
    validate_inputs: Check model inputs for unprocessable values.

Classes:
    DataInputSchema: Pydantic schema for a single data input.
    MultipleDataInputs: Pydantic schema for multiple data inputs.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from log_reg_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Check model inputs for NA values and filter.

    Args:
        input_data (pd.DataFrame): The input DataFrame containing model features.

    Returns:
        pd.DataFrame: The filtered DataFrame with NA values removed.
    """
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Check model inputs for unprocessable values.

    Args:
        input_data (pd.DataFrame): The input DataFrame containing model features.

    Returns:
        Tuple[pd.DataFrame, Optional[dict]]: A tuple containing the validated
        DataFrame and any validation errors.
    """
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # Replace numpy nans so that Pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    """
    Pydantic schema for a single data input.

    Each attribute represents a feature in the input data.
    """
    radius_mean: Optional[float]
    texture_mean: Optional[float]
    perimeter_mean: Optional[float]
    area_mean: Optional[float]
    smoothness_mean: Optional[float]
    compactness_mean: Optional[float]
    concavity_mean: Optional[float]
    concave_points_mean: Optional[float]
    symmetry_mean: Optional[float]
    fractal_dimension_mean: Optional[float]
    radius_se: Optional[float]
    texture_se: Optional[float]
    perimeter_se: Optional[float]
    area_se: Optional[float]
    smoothness_se: Optional[float]
    compactness_se: Optional[float]
    concavity_se: Optional[float]
    concave_points_se: Optional[float]
    symmetry_se: Optional[float]
    fractal_dimension_se: Optional[float]
    radius_worst: Optional[float]
    texture_worst: Optional[float]
    perimeter_worst: Optional[float]
    area_worst: Optional[float]
    smoothness_worst: Optional[float]
    compactness_worst: Optional[float]
    concavity_worst: Optional[float]
    concave_points_worst: Optional[float]
    symmetry_worst: Optional[float]
    fractal_dimension_worst: Optional[float]


class MultipleDataInputs(BaseModel):
    """
    Pydantic schema for multiple data inputs.

    Each input should adhere to the DataInputSchema.

    Attributes:
        inputs (List[DataInputSchema]): A list of data inputs.
    """
    inputs: List[DataInputSchema]
