import numpy as np
import pandas as pd
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
