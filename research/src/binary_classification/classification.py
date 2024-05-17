"""
Implements all methods for training, optimization, evaluation and prediction
using ML models defined in Model for binary classification with numerical scaled features.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.utils.utils as utils

from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn import metrics


def plot_learning_curve(model_obj, X, y, cv=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plots learning curve of a model training to asses if it is
    underfitting or overfitting using cross-validation.
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
    Function that performs hyperparameter optimization for a given model.
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
    Given target real values and predicted ones by a model, this function
    calculates different classification metrics and return them in a dict.
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
    Function to train model and store its weights. 
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

    confusion_matrix = metrics.confusion_matrix(y_test, preds_test)
    class_labels = ['Class 0', 'Class 1']

    # Create a heatmap plot of the confusion matrix
    utils.plot_confusion_matrix(confusion_matrix, class_labels)

    if save_outputs:
        utils.save_model_outputs(model, model_obj.model_name, model_obj.hyperparameters,
                                 eval_train, eval_test, output_dir)
