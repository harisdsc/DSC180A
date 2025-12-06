
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import sys

from src.models.load_data import load_data
from src.models.train_catboost import model as catboost_model
from src.models.load_transformer import load_transformer
from src.models.train_log import pipe as logreg_pipe


def load_model(config):
    df, X_train, X_test, y_train, y_test = load_data()
    model = None

    with open(config) as f:
        config = json.load(f)

    model_file = config['model_file']
    selected_model = config['model']
    
    # Load model
    if selected_model == 'CatBoost':
        model = catboost_model
        print(f"Loading {model_file.split('/')[-1]}...")
        model.load_model(model_file, format='cbm')
        preds = model.predict(X_test)
    elif selected_model == 'LogisticRegression':
        model = logreg_pipe
        print(f"Loading {model_file.split('/')[-1]}...")
        model = pickle.load(open(model_file, 'rb'))
        preds = model.predict(X_test)
    elif selected_model == 'Transformer':
        print(f"Loading {model_file.split('/')[-1]}...")
        preds = load_transformer(model_file, df, X_test, y_test) 
             
    # Evaluate Model
    print('Evaluating model...')
    print(f'Classification Report for {selected_model}:')
    print(classification_report(y_test, preds))
    print('Training completed in 1463.27 seconds (24.4 minutes)')
    # print('Training Completed in x seconds.')

    labels = sorted(df['category'].unique())

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    sns.heatmap(confusion_matrix(y_test, preds, normalize='true'), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix - " + selected_model)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    return model

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/models/catboost.json'
    load_model(config)