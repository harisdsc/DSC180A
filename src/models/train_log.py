from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import time

text_col = 'clean_memo'
cat_cols = ['day_of_week', 'month', 'quarter', 'whole_dollar', 'prev_holiday', \
            'next_holiday']
num_cols = ['amount',
            'day_of_month',
            'dow_sin',
            'dow_cos',
            'month_sin',
            'month_cos',
            'quarter_sin',
            'quarter_cos',
            'log_amount',
            'cents',
            'days_since_last_txn',
            'user_memo_count',
            'days_since_last_txn_z',
            'user_memo_count_z',
            'year',
            'days_since_prev',
            'avg_days_between_txn',
            'rolling_avg_days_between_txn',
            'days_since_first_txn',
            'month_med_amnt',
            'month_med_amnt_diff',
            'log_amnt',
            'days_since_prev_holiday',
            'days_until_next_holiday']

preprocess = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 5)), text_col),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, verbose=1))
])

def train_logistic_regression(X_train, y_train, X_test, y_test, model_file):
    print("Training Logistic Regression...")
    start = time.time()
    pipe.fit(X_train, y_train)

    train_end = time.time() - start
    print(f'Training completed in {train_end:.2f} seconds.')
    
    # Save Model
    print('Saving Model...')
    pickle.dump(pipe, open(model_file, 'wb'))

    return pipe