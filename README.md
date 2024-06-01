# BreastCancerPrediction

The primary objective of this self-learning project is to evaluate various classic machine learning algorithms for binary classification using the [Breast Cancer Prediction dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). Subsequently, the goal is to develop a Python package for deploying the best-performing model found.

Furthermore, this project aims to highlight my programming skills and demonstrate my ability to develop a comprehensive project from the research phase through to model deployment, showcasing my proficiency in delivering production-ready code.

The second phase of this project includes serving the model via a REST API. The app development can be found in the [BreastCancerPredictionAPI repo](https://github.com/mcollpol/BreastCancerPredictionAPI).

# Installation

The package can be installed by: 

```bash
pip install "git+https://github.com/mcollpol/BreastCancerPrediction.git@v0.0.1#egg=mcp-binnary-classification-model&subdirectory=production"
```

Note: Update the package version if needed.

To import its functionality:
```bash
import log_reg_model
```
# Using Tox

Tox is a tool for automating and managing testing environments in Python projects. It helps ensure consistent behavior across different Python versions and environments.

## Prerequisites

Make sure you have Python and Tox installed on your system. You can install Tox using pip:

```bash
pip install tox
```
Note: Expected Tox version 4.

## Usage

### Training the Model

To train the model, use the following command:
```bash
tox -e train
```
This command will call the train_pipeline.py script to generate a trained model.

### Running Tests

To run unit tests and code quality checks, use the following command:

```bash
tox -e test_package
```

For checking only code quality, use the following command:

```bash
tox -e checks
```

