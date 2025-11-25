import numpy as np
import pandas as pd
import time
import json
import sys
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.feature_extraction.date_amnt_feats import create_date_feats, create_amnt_feats

def load_data():
    # Load Data
    print('Loading data...')
    df = pd.read_parquet('data/outflows.pqt')
    memos = pd.read_csv('data/memos_clean.csv')
    df['clean_memo'] = memos['clean_memo']
    
    df['clean_memo'] = df['clean_memo'].fillna('UNKNOWN')
    df = df[df['memo'] != df['category']]
    
    # Feature Engineering
    print('Creating features..')
    df['posted_date'] = pd.to_datetime(df['posted_date'])
    df['day_of_month'] = df['posted_date'].dt.day
    df['day_of_week'] = df['posted_date'].dt.dayofweek
    df['month'] = df['posted_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['log_amount'] = np.log1p(df['amount']) 
    df['cents'] = (df['amount'] * 100) % 100 
    df['whole_dollar'] = np.where(df['cents'] == 0, 1, 0)
    df = df.sort_values(['prism_consumer_id', 'posted_date'])
    df['days_since_last_txn'] = df.groupby('prism_consumer_id')['posted_date'].diff().dt.days
    df['user_memo_count'] = df.groupby(['prism_consumer_id', 'clean_memo']).cumcount()
    df = create_date_feats(df)
    df = create_amnt_feats(df)
    
    return df
    
def train_model(config):
    with open(config) as f:
        config = json.load(f)

    output_file = config['output_file']

    df = load_data()
    
    # Split data
    X = df.drop(columns=['posted_date', 'category', 'memo'])
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    text_cols = ['clean_memo']
    cat_cols = ['day_of_week', 'month', 'quarter', 'whole_dollar', 'prism_consumer_id', 'prism_account_id']    
    
    # Initialize Model
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

    # Train CatBoost
    print("Training CatBoost Model...")
    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        plot=False 
    )

    train_end = time.time() - train_start
    print(f'Training completed in {train_end:.2f} seconds.')
    
    # Evaluate Model 
    print('Evaluating Model...')
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    
    # Save Model
    print('Saving Model...')
    model.save_model(output_file)

def load_model():
    with open(config) as f:
        config = json.load(f)

    catboost_file = config['catboost_file']

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

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        config = args[1]
        if args[1] == 'load':
            load_model()
        elif args[1] == 'train':
            train_model():
        else:
            
    config = args[1] if len(args) > 1 else 'configs/models/train.json'
    train_model(config)
    