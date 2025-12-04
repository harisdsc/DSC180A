import numpy as np
import pandas as pd
from src.feature_extraction.date_amnt_feats import create_date_feats, create_amnt_feats
import sys
import os
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(repo_root)
from src.feature_extraction.holiday_feats import add_custom_holidays, holiday_context, generate_holiday_features
import holidays
from datetime import timedelta
from pandarallel import pandarallel

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
    df = create_date_feats(df)
    df = create_amnt_feats(df)
    pandarallel.initialize(progress_bar=False, verbose=0)
    df = generate_holiday_features(df)
    return df