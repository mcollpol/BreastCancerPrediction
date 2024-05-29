"""
Unit testing for custom sklearn Transformers for data preprocessing.

This module includes custom transformers that can be integrated into an
sklearn pipeline for preprocessing tasks, specifically for detecting and
handling outliers, and applying square root transformations to specified variables.

Classes:
    OutliersTransformer: Detects outliers in specified variables and applies winsorizing to them.
    SquareRootTransformer: Applies square root transformation to specified variables.

Raises:
    ValueError: If the input parameters are not of the expected type or if the
    specified variables are not found in the input DataFrame.
"""

import numpy as np

from log_reg_model.processing.features import OutliersTransformer, SquareRootTransformer


def test_outliers_transformer(example_data_with_outliers):
    """
    Test case for OutliersTransformer.

    Args:
        example_data_with_outliers (pd.DataFrame): Example DataFrame with outliers.
    """
    transformer = OutliersTransformer(variables=['A', 'B'],
                                      threshold=3,
                                      limits=(0.05, 0.05))
    transformer.fit(example_data_with_outliers)

    assert transformer.outliers.loc[0, 'A'] is True
    assert transformer.outliers.loc[1, 'B'] is True

    transformed_data = transformer.transform(example_data_with_outliers)

    assert transformed_data.loc[0, 'A'] != 1000
    assert transformed_data.loc[1, 'B'] != -1000


def test_square_root_transformer(example_data_with_outliers):
    """
    Test case for SquareRootTransformer.

    Args:
        example_data (pd.DataFrame): Example DataFrame.
    """
    transformer = SquareRootTransformer(variables=['A', 'B', 'C'])
    transformed_data = transformer.fit_transform(example_data_with_outliers)

    assert np.allclose(transformed_data['A'].iloc[0],
                       np.sqrt(example_data_with_outliers['A'].iloc[0]))
    assert np.allclose(transformed_data['B'].iloc[1],
                       np.sqrt(example_data_with_outliers['B'].iloc[1]))
    assert np.allclose(transformed_data['C'],
                       np.sqrt(example_data_with_outliers['C']))
