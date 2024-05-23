"""
Implements all methods for training, optimization, evaluation and prediction
using ML models defined in Model for binary classification with numerical scaled features.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn import metrics

from src.utils import utils


def plot_learning_curve(model_obj, X, y, cv=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plots learning curve of a model training to asses if it is
    underfitting or overfitting using cross-validation.

    Parameters:
    ----------
    model_obj : object
        The model object containing the model to be evaluated.
        It should have an attribute `model` which is the actual model instance.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features
        is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_outputs)
        Target relative to X for classification or regression.

    cv : int, cross-validation generator or an iterable, optional (default=None)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined by the
        selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to be big
        enough to contain at least one sample from each class.

    Returns:
    -------
    None
        This function does not return anything. It displays a plot of the learning curve.
    """
    model = model_obj.model

    train_sizes, train_scores, val_scores = learning_curve(model,
                                                           X,
                                                           y,
                                                           cv=cv,
                                                           train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation Score')

    plt.legend(loc='best')
    plt.show()


def hyperparameter_optimization(model, param_grid, X, y, cv=5):
    """
    Performs hyperparameter optimization for a given model using grid
    search with cross-validation.

    Parameters:
    ----------
    model : estimator object
        The object of the model for which the hyperparameters need to be optimized.
        This object must implement the `fit` and `predict` methods.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of parameter settings
        to try as values, or a list of such dictionaries, in which case the grids spanned
        by each dictionary in the list are explored. This enables searching over any sequence
        of parameter settings.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number
        of features.

    y : array-like, shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression.

    cv : int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds in a (Stratified)KFold,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

    Returns:
    -------
    best_params : dict
        Parameter setting that gave the best results on the hold out data.

    Notes:
    -----
    Prints the best hyperparameters and the corresponding score.
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f'Best hyperparameters found: {best_params}')
    print(f'Best score: {round(best_score, 4)}')

    return best_params


def evaluation(y_true, y_pred):
    """
    Calculates various classification metrics given the true and predicted
    target values, and returns them in a dictionary.

    Parameters:
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.

    y_pred : array-like, shape (n_samples,)
        Predicted target values by the model.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the calculated classification metrics:
        - accuracy
        - precision
        - recall
        - f1 score
        - AUC-ROC score

    Notes:
    -----
    The function rounds each metric to 4 decimal places.
    """
    results = {}

    results['accuracy'] = [round(metrics.accuracy_score(y_true, y_pred), 4)]
    results['precision'] = [round(metrics.precision_score(y_true, y_pred), 4)]
    results['recall'] = [round(metrics.recall_score(y_true, y_pred), 4)]
    results['f1'] = [round(metrics.f1_score(y_true, y_pred), 4)]
    results['auc_roc'] = [round(metrics.roc_auc_score(y_true, y_pred), 4)]

    return pd.DataFrame(results)


def train_and_evaluate_model(model_obj, X_train, y_train, X_test, y_test,
                             save_outputs=False, output_dir = ""):
    """
    Trains a model, evaluates its performance on training and test sets,
    and optionally saves the outputs.

    Parameters:
    ----------
    model_obj : object
        An object containing the model to be trained. This object should have the
        following attributes:
        - model: The actual model instance that implements the `fit` and `predict` methods.
        - model_name: A string representing the name of the model.
        - hyperparameters: A dictionary of the model's hyperparameters.

    X_train : array-like, shape (n_samples_train, n_features)
        Training data.

    y_train : array-like, shape (n_samples_train,)
        Training labels.

    X_test : array-like, shape (n_samples_test, n_features)
        Test data.

    y_test : array-like, shape (n_samples_test,)
        Test labels.

    save_outputs : bool, optional (default=False)
        If True, the model and evaluation outputs will be saved
        to the specified output directory.

    output_dir : str, optional (default="")
        The directory where the model and evaluation outputs will be saved
        if `save_outputs` is True.

    Returns:
    -------
    None
        This function does not return anything. It prints evaluation results
        and optionally saves the outputs.

    Notes:
    -----
    The function also plots a confusion matrix for the test set predictions.
    """

    model = model_obj.model
    model.fit(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    eval_train = evaluation(y_train, preds_train)
    eval_test = evaluation(y_test, preds_test)

    print(f"For model {model_obj.model_name} with {model_obj.hyperparameters}:\n")
    print("Results for train set:\n")
    print(eval_train)

    print("\nResults for test set:\n")
    print(eval_test)

    confusion_matrix = metrics.confusion_matrix(y_test,
                                                preds_test)
    class_labels = ['Class 0', 'Class 1']

    # Create a heatmap plot of the confusion matrix
    utils.plot_confusion_matrix(confusion_matrix,
                                class_labels)

    if save_outputs:
        utils.save_model_outputs(model_obj,
                                 eval_train,
                                 eval_test,
                                 output_dir)
