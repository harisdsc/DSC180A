def train_model(config):
    with open(config) as f:
        config = json.load(f)

    output_file = config['output_file']

    df = load_data()

    X = df.drop(columns=['posted_date', 'category', 'memo'])
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    text_col = "cleaned_memo"

    cat_cols = ["day_of_week", "month", "quarter", "year", "whole_dollar", "prev_holiday", "next_holiday"]
    num_cols = [
        "days_since_prev",
        "avg_days_between_txn",
        "rolling_avg_days_between_txn",
        "days_since_first_txn",
        "month_med_amnt",
        "month_med_amnt_diff",
        "amnt_zscore",
        "log_amnt",
        "days_since_prev_holiday",
        "days_until_next_holiday"
    ]
    X_train = outflow_train
    y_train = outflow_train['category']
    X_test = outflow_test
    y_test = outflow_test['category']
    # -------------------------
    # Create Preprocessing
    # -------------------------
    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 5)), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # -------------------------
    # Full Pipeline
    # -------------------------
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    # -------------------------
    # Train
    # -------------------------
    start = time.time()
    pipe.fit(X_train, y_train)
    train_end = time.time() - start
    print(f'Training completed in {train_end:.2f} seconds.')
    
    # -------------------------
    # Predict
    # -------------------------
    print('Evaluating Model...')
    y_pred = pipe.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model
    print('Saving Model...')
    model.save_model(output_file)

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/models/train.json'
    train_model(config)