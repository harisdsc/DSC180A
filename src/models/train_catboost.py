from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
import time


text_cols = ['clean_memo']
cat_cols = ['day_of_week', 'month', 'quarter', 'whole_dollar', 'prev_holiday', \
            'next_holiday']

# Initialize Model
model = CatBoostClassifier(
    iterations=10_000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=1000,
    early_stopping_rounds=100,
    task_type='GPU' if get_gpu_device_count() > 0 else 'CPU',
    devices='0:1' if get_gpu_device_count() > 1 else None,
    cat_features=cat_cols, 
    text_features=text_cols,
    # auto_class_weights='Balanced',
    # text_processing=["NaiveBayes+Word,BiGram|BoW+Word,BiGram"]
)

def train_catboost(X_train, y_train, X_test, y_test, model_file):
        # Train CatBoost
        print("Training CatBoost...")
        train_start = time.time()
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False 
        )

        train_end = time.time() - train_start
        print(f'Training completed in {train_end:.2f} seconds.')

        # Save Model
        print('Saving model...')
        model.save_model(model_file)

        return model
        
        