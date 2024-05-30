"""
Module for defining pytest fixtures for testing.

This module defines pytest fixtures that can be used for testing purposes.
"""
import numpy as np
import pandas as pd
import pytest

from log_reg_model.config.core import config
from log_reg_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    """
    Fixture for providing sample input data for testing.

    This fixture loads sample input data from a test data file specified in the configuration.
    Returns:
        pd.DataFrame: The sample input data.
    """
    return load_dataset(file_name=config.app_config.test_data_file)


@pytest.fixture
def test_sample_data():
    """
    Fixture for generating example DataFrame with some outliers.

    Returns:
        pd.DataFrame: Example DataFrame with outliers.
    """
    np.random.seed(0)
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100),
        'C': np.random.normal(0, 1, 100),
    })

    # Creating outliers.
    data.loc[0, 'A'] = 1000
    data.loc[1, 'B'] = -1000

    # Ensure that 'C' column is always positive.
    data['C'] = np.abs(data['C'])

    return data
