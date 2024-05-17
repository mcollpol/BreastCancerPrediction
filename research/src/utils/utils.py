"""
Module to implement utility functions.
"""
import os
import pickle
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(data_path):
    """
    Loads cleaned data from csv. Returns pandas DataFrame.
    """
    assert os.path.exists(data_path), f"Filepath {data_path} does not exist"
    assert os.path.splitext(data_path)[1] == ".csv", "File is not a CSV"

    df = pd.read_csv(data_path)

    return df


def save_dict_to_csv(data_dict, data_path):
    """
    Save dict() to csv.
    """
    with open(data_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)


def save_model_outputs(model, model_name, hyperparams, eval_train, eval_test, output_dir):
    """
    Saves model to a pickle, model hyperparams and evaluation results to output_dir.
    """
    # Create directories if they don't exist.
    model_dir = output_dir + '/' + model_name
    os.makedirs(model_dir, exist_ok=True)

    # Save the model.
    with open(f'{model_dir}/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save best params.
    save_dict_to_csv(hyperparams, f'{model_dir}/params.csv' )

    # Save the evaluation.
    eval_train.to_csv(f'{model_dir}/Train_Evaluation.csv')
    eval_test.to_csv(f'{model_dir}/Test_Evaluation.csv')


def get_method_name(name):
    """
    Converts MethodName into method_name.
    """
    return 'define_' + ''.join(['_' + c.lower() if c.isupper()
                                else c for c in name]).lstrip('_')


def transform_param_ranges(param_grid, model_name, param_ranges):
    """
    Transforms parameters to optimize with GridSearch for a model and
    ads them to existing param grid.
    Meant to be used for models that combine classifiers like ensamble models.  
    """
    transformed_ranges = {}
    for param, values in param_ranges.items():
        transformed_param = f'{model_name}__{param}'
        transformed_ranges[transformed_param] = values
    return param_grid.update(transformed_ranges)


def plot_confusion_matrix(confusion_matrix, class_labels):
    """
    Plots a confusion matrix using a headmap.
    """
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)

    # Add labels, title, and ticks
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Test set')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)

