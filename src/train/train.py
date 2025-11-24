import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load Data
df = pd.read_parquet('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt')
memos = pd.read_csv('../../data/memos_cleaned.csv')
df['clean_memo'] = memos['clean_memo']

# Feature Engineering
df['posted_date'] = pd.to_datetime(df['posted_date'])
df['day_of_month'] = df['posted_date'].dt.day
df['day_of_week'] = df['posted_date'].dt.dayofweek
df['month'] = df['posted_date'].dt.month
df['clean_memo'] = df['clean_memo'].fillna("UNKNOWN")

X = df.drop(columns=['prism_consumer_id', 'prism_account_id', 'posted_date', 'category', 'memo'])
y = df['category']

# Define Feature Types
text_features = ['clean_memo']
categorical_features = []
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize & Train Catboost
print("Training CatBoost Model...")

model = CatBoostClassifier(
    iterations=500,             # Number of trees (increase to 1000+ for large data)
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass', # Detects multiclass automatically
    eval_metric='Accuracy',
    verbose=100,                # Log output every 100 trees
    early_stopping_rounds=50,   # Stop if validation score stops improving
)

model.fit(
    X_train, y_train,
    text_features=text_features,
    eval_set=(X_test, y_test),
    plot=False
)

# Evaluate Model 
print("\n--- Evaluation ---")
preds = model.predict(X_test)
print(classification_report(y_test, preds))
