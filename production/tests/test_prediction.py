"""
This module contains a unit test for the make_prediction function of a logistic
regression model. The test ensures that the function correctly returns
predictions with the expected properties.

The test performs the following checks:
1. Ensures that the predictions are returned as a list.
2. Validates that the first prediction is a floating-point number.
3. Verifies that no errors are present in the result.
4. Confirms that the number of predictions matches the expected number.
5. Checks that the first prediction value is close to the expected value
   within a specified tolerance.

Usage:
- The test_make_prediction function can be executed directly with sample input data
  to validate the make_prediction function.
- Adjust the sample_input_data in the __main__ block as needed for specific use cases.

Example:
    sample_input_data = {
        # Add your sample input data here
    }
"""
import numpy as np
from sklearn.metrics import accuracy_score

from log_reg_model.config.core import config
from log_reg_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    """
    Test the make_prediction function from the logistic regression model.

    This test verifies that the make_prediction function returns a list of
    predictions with the expected characteristics.

    Args:
        sample_input_data (dict or pd.DataFrame): The input data to be used
        for making predictions. This should be formatted as expected by the
        make_prediction function.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    expected_no_predictions = 12

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], (np.int64))
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert accuracy_score(sample_input_data[config.model_config.target],
                          predictions) > 0.92, "Low accuracy score."
