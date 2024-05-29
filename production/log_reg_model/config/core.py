"""
This module sets up the configuration for the logistic regression model training project.

It includes:
- Definition of project directories.
- Configuration classes for application-level and model-specific settings using Pydantic.
- Functions to locate, load, parse, and validate the configuration from a YAML file.

Classes:
    AppConfig: Application-level configuration.
    ModelConfig: Model-specific configuration.
    Config: Master configuration object that includes both AppConfig and ModelConfig.

Functions:
    find_config_file: Locate the configuration file.
    fetch_config_from_yaml: Parse YAML containing the package configuration.
    create_and_validate_config: Run validation on configuration values.

Usage:
    This module is typically used to load and validate the configuration settings for the project,
    ensuring all necessary parameters are correctly specified before running the model training pipeline.

Example:
    config = create_and_validate_config()

Raises:
    FileNotFoundError: If the configuration file is not found.
    OSError: If there is an issue reading the configuration file.
    ValidationError: If the configuration data does not match the expected schema.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import log_reg_model


# Project Directories
PACKAGE_ROOT = Path(log_reg_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level configuration.

    Attributes:
        package_name (str): Name of the package.
        data_file (str): Path to the data file.
        pipeline_name (str): Name of the pipeline.
        pipeline_save_file (str): Path where the pipeline save file will be stored.
        columns_to_drop (List[str]): List of columns to drop from the dataset.
    """
    package_name: str
    data_file: str
    pipeline_name: str
    pipeline_save_file: str
    columns_to_drop: List[str]

class ModelConfig(BaseModel):
    """
    Configuration relevant to model training and feature engineering.

    Attributes:
        target (str): Name of the target variable.
        features (List[str]): List of feature names.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        outliers_threshold (int): Threshold for detecting outliers.
        outliers_limits (float): Limits for outlier detection.
        scaler_feature_range (Sequence[int]): Feature range for the scaler (min, max).
        square_root_vars (Sequence[str]): Variables to transform using square root.
        log_transform_var (Sequence[str]): Variable to transform using logarithm.
        model_params (Dict[str, Any]): Parameters for the model.
    """
    target: str
    features: List[str]
    test_size: float
    random_state: int
    outliers_threshold: int
    outliers_limits: float
    scaler_feature_range: Sequence[int]  # Tuple.
    square_root_vars: Sequence[str]
    log_transform_var: Sequence[str]
    model_params: Dict[str, Any]

class Config(BaseModel):
    """
    Master configuration object that includes both application and model configurations.

    Attributes:
        app_config (AppConfig): Application-level configuration.
        model_config (ModelConfig): Model-specific configuration.
    """
    app_config: AppConfig
    model_config: ModelConfig

def find_config_file() -> Path:
    """
    Locate the configuration file.

    Returns:
        Path: Path to the configuration file.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """
    Parse YAML containing the package configuration.

    Args:
        cfg_path (Optional[Path]): Optional path to the YAML configuration file.

    Returns:
        YAML: Parsed YAML configuration.

    Raises:
        OSError: If the configuration file is not found at the specified path.
    """
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r", encoding='utf-8') as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """
    Run validation on configuration values.

    Args:
        parsed_config (YAML): Parsed YAML configuration.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValidationError: If the configuration data does not match the expected schema.
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config

# Load and validate the configuration.
config = create_and_validate_config()
