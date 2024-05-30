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


def test_outliers_transformer(test_sample_data):
    """
    Test case for OutliersTransformer.

    Args:
        example_data_with_outliers (pd.DataFrame): Example DataFrame with outliers.
    """
    transformer = OutliersTransformer(variables=['A', 'B'],
                                      threshold=3,
                                      limits=(0.1, 0.1))
    transformer.fit(test_sample_data)

    assert transformer.outliers.loc[0, 'A'], "Outlier not detected in column 'A'"
    assert transformer.outliers.loc[1, 'B'], "Outlier not detected in column 'B'"

    transformed_data = transformer.transform(test_sample_data)

    assert transformed_data.loc[0, 'A'] != 1000, "Transformation has not been applied to outlier"
    assert transformed_data.loc[1, 'B'] != -1000, "Transformation has not been applied to outlier"


def test_square_root_transformer(test_sample_data):
    """
    Test case for SquareRootTransformer.

    Args:
        example_data (pd.DataFrame): Example DataFrame.
    """
    transformer = SquareRootTransformer(variables=['C'])
    transformed_data = transformer.fit_transform(test_sample_data)

    expected_result = np.sqrt(test_sample_data['C'])

    np.testing.assert_array_almost_equal(transformed_data['C'], expected_result)
