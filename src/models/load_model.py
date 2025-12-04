import numpy as np
import pandas as pd
import json
import sys
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.models.load_data import load_data


def load_model(config):
    with open(config) as f:
        config = json.load(f)

    input_file = config['input_file']

    df = load_data()
    
    # Split data
    X = df.drop(columns=['posted_date', 'category', 'memo'])
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    text_cols = ['clean_memo']
    cat_cols = ['day_of_week', 'month', 'quarter', 'whole_dollar', 'prism_consumer_id', 'prism_account_id']    
    
    # Initialize Catboost
    model = CatBoostClassifier(
        iterations=10_000,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        # auto_class_weights='Balanced',
        verbose=1000,
        early_stopping_rounds=100,
        task_type='GPU',
        devices='0:1',
        cat_features=cat_cols, 
        text_features=text_cols,
        # text_processing=["NaiveBayes+Word,BiGram|BoW+Word,BiGram"]
    )

    # Load Model
    print("Loading CatBoost Model...")
    model.load_model(input_file, format='cbm')

    # Evaluate Model
    print('Evaluating Model...')
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))
        

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/models/load.json'
    load_model(config)