"""
Module to implement utility functions.
"""
import os
import pickle
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def save_dict_to_csv(data_dict, data_path):
    """
    Save a dictionary to a CSV file.

    Parameters:
    ----------
    data_dict : dict
        The dictionary to be saved to the CSV file.

    data_path : str
        The path to save the CSV file.

    Returns:
    -------
    None

    Raises:
    -------
    IOError:
        If an error occurs while writing to the CSV file.

    Notes:
    ------
    This function saves the contents of a dictionary to a CSV file. Each key-value pair
    in the dictionary corresponds to a row in the CSV file, where the keys are the column
    headers and the values are the values in the row.
    """
    try:
        with open(data_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data_dict.keys())
            writer.writeheader()
            writer.writerow(data_dict)
    except IOError as e:
        print(f"An error occurred while writing to the CSV file: {e}")
        raise


def save_model_outputs(model_obj, eval_train, eval_test, output_dir):
    """
    Save model outputs to specified directory.

    Parameters:
    ----------
    model : object
        The trained model object to be saved.

    eval_train : DataFrame
        Evaluation metrics for the training set.

    eval_test : DataFrame
        Evaluation metrics for the test set.

    output_dir : str
        The directory path where outputs will be saved.

    Returns:
    -------
    None

    Raises:
    -------
    IOError:
        If an error occurs while writing to files.

    Notes:
    ------
    This function saves the trained model, its hyperparameters, and evaluation results
    to the specified output directory. It creates subdirectories for each model if they
    do not exist.
    """
    try:
        # Create model directory if it doesn't exist.
        model_dir = os.path.join(output_dir, model_obj.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model.
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as file:
            pickle.dump(model_obj.model, file)

        # Save best parameters.
        save_dict_to_csv(model_obj.hyperparameters, os.path.join(model_dir, 'params.csv'))

        # Save the evaluation.
        eval_train.to_csv(os.path.join(model_dir, 'Train_Evaluation.csv'), index=False)
        eval_test.to_csv(os.path.join(model_dir, 'Test_Evaluation.csv'), index=False)
    except IOError as e:
        print(f"An error occurred while writing model outputs: {e}")
        raise


def get_method_name(name):
    """
    Converts a camel case method name into a snake case method name prefixed with 'define_'.

    Args:
        name (str): The camel case method name to be converted.

    Returns:
        str: The converted snake case method name prefixed with 'define_'.
    """
    return 'define_' + ''.join(['_' + c.lower() if c.isupper()
                                else c for c in name]).lstrip('_')


def transform_param_ranges(param_grid, model_name, param_ranges):
    """
    Transforms parameter names to be compatible with GridSearchCV for a given model
    and adds them to the existing parameter grid.

    This function is particularly useful for ensemble models that combine multiple classifiers.

    Args:
        param_grid (dict): The existing parameter grid to be updated.
        model_name (str): The name of the model to be prefixed to each parameter.
        param_ranges (dict): The parameter ranges to be transformed and added to the grid.

    Returns:
        None: The param_grid is updated in place.
    """
    transformed_ranges = {}
    for param, values in param_ranges.items():
        transformed_param = f'{model_name}__{param}'
        transformed_ranges[transformed_param] = values
    return param_grid.update(transformed_ranges)


def plot_confusion_matrix(confusion_matrix, class_labels):
    """
    Plots a confusion matrix using a heatmap.

    Args:
        confusion_matrix (ndarray): The confusion matrix to be plotted.
        class_labels (list): The list of class labels to be used for the x and y axes.

    Returns:
        None: The function creates a plot of the confusion matrix.
    """
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    # Add labels, title, and ticks
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Test set')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)

