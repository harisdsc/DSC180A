import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.feature_extraction.date_amnt_feats import create_date_feats, create_amnt_feats

# Load Data
df = pd.read_parquet('data/outflows.pqt')
memos = pd.read_csv('data/memos_clean.csv')
df['clean_memo'] = memos['clean_memo']

df['clean_memo'] = df['clean_memo'].fillna('UNKNOWN')
df = df[df['memo'] != df['category']]

# Feature Engineering
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

X = df.drop(columns=['posted_date', 'category', 'memo'])
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Feature Types
text_cols = ['clean_memo']
cat_cols = ['day_of_week', 'month', 'quarter', 'whole_dollar', 'prism_consumer_id', 'prism_account_id']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize & Train Catboost
print("Training CatBoost Model...")

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

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    plot=False 
)

# Evaluate Model 
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Save Model
model.save_model("catboost.cbm")
