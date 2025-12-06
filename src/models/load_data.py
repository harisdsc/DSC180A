import numpy as np
import pandas as pd
import subprocess
import time
import os
from sklearn.model_selection import train_test_split

from src.preprocessing.rules import TransactionCleaner
from src.feature_extraction.date_amnt_feats import create_date_feats, create_amnt_feats
from src.feature_extraction.holiday_feats import generate_holiday_features

def load_data(script=False):
    # Load Data
    print('Processing data...')
    if not script and os.path.exists('data/outflows_clean.csv'):
        print('Loading cached processed data...')
        df = pd.read_csv('data/outflows_clean.csv')
    else:
        if os.path.exists('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt'):
            df = pd.read_parquet('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt')
        else:
            df = pd.read_parquet('data/outflows.pqt')

        df = df[df['memo'] != df['category']]

        if os.path.exists('data/memo_clean.csv') and not script:
            memos = pd.read_csv('data/memo_clean.csv')
            df['clean_memo'] = memos['clean_memo']
        else:
            print('Cleaning memos...')
            clean_start = time.time()
            cleaner = TransactionCleaner()
            df['clean_memo'] = df['memo'].apply(cleaner.clean)
            print(f'Cleaning completed in {time.time() - clean_start:.2f} seconds.')
            print('Saving cleaned memos...')
            df['clean_memo'].to_csv('data/memo_clean.csv', index=False)
            
        # Feature Engineering
        print('Creating features...')
        start_feats = time.time()
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        df['day_of_month'] = df['posted_date'].dt.day
        df['day_of_week'] = df['posted_date'].dt.dayofweek
        df['month'] = df['posted_date'].dt.month
        df["quarter"] = df['posted_date'].dt.quarter
        df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"]   / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"]   / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
        df['log_amount'] = np.log1p(df['amount']) 
        df['cents'] = (df['amount'] * 100) % 100 
        df['whole_dollar'] = np.where(df['cents'] == 0, 1, 0)
        df = df.sort_values(['prism_consumer_id', 'posted_date'])
        df['days_since_last_txn'] = df.groupby('prism_consumer_id')['posted_date'].diff().dt.days
        df['days_since_last_txn'] = df['days_since_last_txn'].fillna(0)
        df['user_memo_count'] = df.groupby(['prism_consumer_id', 'clean_memo']).cumcount()
        df['user_memo_count'] = df['user_memo_count'].fillna(0)
        df['days_since_last_txn_z'] = np.log1p(df['days_since_last_txn'])
        df['user_memo_count_z'] = np.log1p(df['user_memo_count'])

        df = create_date_feats(df)
        df = create_amnt_feats(df)
        df = generate_holiday_features(df)

        print(f'Feature engineering completed in {time.time() - start_feats:.2f} seconds.')

        print('Saving processed data...')
        df.to_csv('data/outflows_clean.csv', index=False)

    df['clean_memo'] = df['clean_memo'].fillna(df['memo'])

    outflow_consumers = df["prism_consumer_id"].unique()
    train_ids, test_ids = train_test_split(outflow_consumers, test_size=0.2, random_state=42)
    
    train_df = df[df["prism_consumer_id"].isin(train_ids)]
    test_df  = df[df["prism_consumer_id"].isin(test_ids)]

    cols_to_drop = ['posted_date', 'category', 'memo', 'prism_consumer_id', 'prism_account_id']

    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['category']
    
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['category']
    
    return df, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df, X_train, X_test, y_train, y_test = load_data(script=True)