'''
File made for functionality testing. 
'''
import configparser
import os

from utils import utils
from binary_classification import classification as bc
from binary_classification.models import Model


config = configparser.ConfigParser()
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_dir, 'config.ini')
config.read(config_file_path)

DATA_PATH = config.get('LocalSettings', 'DATA_PATH')
OUTPUT_PATH = config.get('LocalSettings', 'OUTPUT_PATH')
TARGET = config.get('LocalSettings', 'TARGET')
PREDICTORS = ['radius_mean', 'texture_mean','perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean']


if __name__ == "__main__":

    # Load and split data.
    df = utils.load_data(DATA_PATH)
    data = utils.split_train_test(df, PREDICTORS, TARGET)

    # Ensamble voting classifier
    estimators = [{'name':'LogisticRegression', 'params':{}},
                  {'name': 'Svm', 'params': {}},
                  {'name':'DecisionTree', 'params':{}}]
    model_obj = Model('EnsambleVoting', estimators_info=estimators)

    bc.plot_learning_curve(model_obj, df[PREDICTORS], df[TARGET], cv=5)
    bc.train_and_evaluate_model(model_obj, data, save_outputs=True, output_dir=OUTPUT_PATH)
