"""
Custom sklearn Transformers for data preprocessing.

"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize


class OutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Detects outliers in specified variables and applies winsorizing to them.

    Parameters:
    -----------
    variables : list
        List of column names containing variables to detect outliers and apply winsorizing.
    threshold : float
        Threshold value for detecting outliers based on z-scores.
    limits : float or tuple
        Winsorizing limits to use for outliers.
        If a float is provided, it will be applied symmetrically to both ends of the distribution.
        If a tuple is provided, it should contain the lower and upper limits separately.

    Attributes:
    -----------
    _outliers : pandas DataFrame
        DataFrame indicating True for outliers and False for non-outliers.
    """

    def __init__(self, variables, threshold=3, limits=(0.05, 0.05)):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        if not isinstance(threshold, (int, float)):
            raise ValueError('threshold should be int or float')
        if not isinstance(limits, (float, tuple)):
            raise ValueError('limits should be a float or a tuple')

        self.variables = variables
        self.threshold = threshold
        self.limits = limits

        self._outliers = None

    def fit(self, X, y=None):
        """
        Fits the transformer by detecting outliers in the specified variables.

        Parameters:
        -----------
        X : pandas DataFrame
            Input DataFrame containing the variables to be transformed.
        y: pandas Series (Optional)
            Target variable. Ensures consistency with sklearn's API conventions.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Checking if all specified variables are present in the DataFrame.
        missing_vars = set(self.variables) - set(X.columns)
        if missing_vars:
            raise ValueError(f"Variables not found in the DataFrame: {missing_vars}")
        self._outliers = pd.DataFrame(index=X.index, columns=self.variables)

        # Calculate z-scores and detect outliers
        for var in self.variables:
            var_std = X[var].std()

            if var_std == 0:  # Handling the case where standard deviation is zero.
                self._outliers[var] = False  # Mark all values as non-outliers.
            else:
                z_scores = (X[var] - X[var].mean()) / var_std
                self._outliers[var] = np.abs(z_scores) > self.threshold

        return self

    def transform(self, X, y=None):
        """
        Transforms the input DataFrame by applying winsorizing to detected outliers.

        Parameters:
        -----------
        X : pandas DataFrame
            Input DataFrame to be transformed.
        y: pandas Series (Optional)
           Target variable. Ensures consistency with sklearn's API conventions.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed DataFrame with winsorized outliers.
        """
        X_transformed = X.copy()

        # Apply winsorizing to detected outliers
        for var in self.variables:
            X_transformed[var] = X_transformed[var].mask(self._outliers[var],
                                                         winsorize(X_transformed[var],
                                                                   limits=self.limits))

        return X_transformed


class SquareRootTransformer(BaseEstimator, TransformerMixin):
    """
    Applies square root transformation to specified variables.

    Parameters:
    -----------
    variables : list
        List of column names containing variables to apply square root transformation.

    Attributes:
    -----------
    None
    """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):
        """
        The fit method is a fundamental requirement for any method
        to seamlessly integrate into the sklearn pipeline.

        Parameters:
        -----------
        X : pandas DataFrame
            Input DataFrame containing the variables to be transformed.
            Even if not used, it ensures consistency with sklearn's API conventions.
        
        y: pandas Series (Optional)
           Target variable. Ensures consistency with sklearn's API conventions.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms the input DataFrame by applying square root
        transformation to specified variables.

        Parameters:
        -----------
        X : pandas DataFrame
            Input DataFrame to be transformed.

        y: pandas Series (Optional)
           Target variable. Ensures consistency with sklearn's API conventions.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed DataFrame with square root transformation
            applied to specified variables.
        """
        missing_vars = set(self.variables) - set(X.columns)
        if missing_vars:
            raise ValueError(f"Variables not found in the DataFrame: {missing_vars}")

        X_transformed = X.copy()

        for var in self.variables:
            X_transformed[var] = X_transformed[var].apply(lambda x: x** 0.5)

        return X_transformed
