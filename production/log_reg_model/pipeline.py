"""
This module defines a data preprocessing and modeling pipeline using scikit-learn and
custom transformers.

The pipeline consists of the following steps:
1. Outlier detection and handling using a custom transformer.
2. Square root transformation on specified variables using a custom transformer.
3. Log transformation on specified variables using a transformer from the Feature-engine package.
4. Feature scaling using MinMaxScaler.
5. Logistic Regression modeling.

The pipeline is configured using parameters defined in the configuration file.

Classes:
    OutliersTransformer: Custom transformer to detect and handle outliers.
    SquareRootTransformer: Custom transformer to apply square root transformation.

Pipeline Steps:
    - outliers_imputation: Applies the OutliersTransformer to handle outliers.
    - square_root: Applies the SquareRootTransformer to specified variables.
    - log: Applies log transformation using LogTransformer from Feature-engine.
    - scaler: Scales features using MinMaxScaler.
    - model: Fits a LogisticRegression model.

Example Usage:
    from sklearn.model_selection import train_test_split
    from log_reg_model.config.core import config
    from log_reg_model.pipeline import pipe

    # Load and split your data
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.features],
        data[config.target],
        test_size=config.test_size,
        random_state=config.random_state)

    # Fit the pipeline
    pipe.fit(X_train, y_train)

    # Predict using the pipeline
    predictions = pipe.predict(X_test)
"""
from feature_engine.transformation import LogTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from log_reg_model.config.core import config
from log_reg_model.processing import features as pp

model_params = config.model_config.model_params

pipe = Pipeline([
    # Outliers.
    ('outliers_imputation', pp.OutliersTransformer(
        variables=config.model_config.features,
        threshold=config.model_config.outliers_threshold,
        limits=config.model_config.outliers_limits
    )),
    # Transformations.
    ('square_root', pp.SquareRootTransformer(variables=config.model_config.square_root_vars)),
    ('log', LogTransformer(variables=config.model_config.log_transform_vars)),
    # Scaling.
    ('scaler', MinMaxScaler(feature_range=config.model_config.scaler_feature_range)),
    # Model.
    ('model', LogisticRegression(
                                penalty=model_params['penalty'],
                                C=float(model_params['C']),
                                solver=model_params['solver'],
                                max_iter=int(model_params['max_iter']),
                                random_state=int(model_params['random_state'])
                                )),
])
