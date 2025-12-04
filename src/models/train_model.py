from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import time
import json
import sys

from src.models.load_data import load_data
from src.models.catboost.catboost import train_catboost
    
def train_model(config):
    with open(config) as f:
        config = json.load(f)

    output_file = config['output_file']
    selected_model = config['model']
    
    # Load Data + Feature Engineering
    df = load_data()
    
    # Split data
    X = df.drop(columns=['posted_date', 'category', 'memo', \
                         'prism_consumer_id', 'prism_account_id'])
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if selected_model == 'Transformer':
            ...
    else:
        if selected_model == 'CatBoost':
            model = train_catboost(X_train, y_train, X_test, y_test, output_file)
            preds = model.predict(X_test)
        elif selected_model == 'LogRegression':
             ...
        
        # Evaluate Model
        print('Evaluating Model...')
        print(classification_report(y_test, preds))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, preds))
        
        
if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/models/train.json'
    train_model(config)
    