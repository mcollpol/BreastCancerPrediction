"""
Implements Feature Selection classes that are sklearn API compatible.
"""
from itertools import zip_longest
from sklearn.feature_selection import SelectFromModel
from sklearn.base import TransformerMixin


class FeatureSelectorEnsemble(TransformerMixin):
    """Select features using an ensemble of models.

    This transformer selects features based on an ensemble of models.
    Each model is used to fit the data and select features.
    The selected features are determined based on the voting threshold.

    Parameters:
    - models (list): List of models to be used for feature selection.
    - max_features_per_model (int or None): Maximum number of features to select per model.
        (None selects all but the ones with 0 impact)
    - voting_threshold (int): Voting threshold for feature selection.

    Attributes:
    - _selected_features (list): List of selected feature names.

    """

    def __init__(self, models, max_features_per_model=10, voting_threshold=2):
        if not isinstance(models, list):
            raise ValueError('models should be a list')
        if not isinstance(max_features_per_model, (int)):
            raise ValueError('max_features_per_model should be an integer')
        if not isinstance(voting_threshold, (int)):
            raise ValueError('voting_threshold should be an ineger')

        self.models = models
        self.max_features_per_model = max_features_per_model
        self.voting_threshold = voting_threshold
        self._selected_features = []

    def fit(self, X, y):
        """Selects features using a voting aproach.

        Parameters:
        - X (DataFrame): Input features.
        - y (Series): Target variable.

        """
        voting_count = []
        for model in self.models:
            model.fit(X, y)
            selector = SelectFromModel(model,
                                       max_features=self.max_features_per_model)
            selector.fit(X, y)
            # Sums two lists element per element ->[1] + [1, 1, 0] =  [2, 1, 0]
            voting_count = [sum(pair)
                            for pair in zip_longest(voting_count,
                                                    selector.get_support().astype(int),
                                                    fillvalue=0)]

        self._selected_features = [col for i, col in enumerate(X.columns)
                                   if voting_count[i] >= self.voting_threshold]
        return self

    def transform(self, X, y=None):
        """Transform the input data by selecting the chosen features.

        Parameters:
        - X (DataFrame): Input features.
        - y (Series): Target variable (unused), here to acomodate sklearn API conventions.

        Returns:
        - (DataFrame): Transformed features with selected columns.

        """
        return X[self._selected_features]
