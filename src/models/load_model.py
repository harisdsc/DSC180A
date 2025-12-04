
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.models.catboost.catboost import model as catboost_model
from torch import torch, TransactionClassifier, TransactionDataset, DataLoader
import json
import sys

from src.models.load_data import load_data

def load_model(config):
    with open(config) as f:
        config = json.load(f)

    model_file = config['model_file']
    selected_model = config['model']

    df = load_data()
    
    # Split data
    X = df.drop(columns=['posted_date', 'category', 'memo', \
                         'prism_consumer_id', 'prism_account_id'])
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    if selected_model == 'CatBoost':
        model = catboost_model
        print(f"Loading CatBoost from {model_file.split('/')[-1]}...")
        model.load_model(model_file, format='cbm')
        preds = model.predict(X_test)
    elif selected_model == 'LogRegression':
        ...
    elif selected_model == 'Transformer':
            print(f"Loading Transformer Model from {model_file.split('/')[-1]}...")
            
            checkpoint = torch.load(model_file)
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

            vocab_map = checkpoint['vocab_map']
            normalization_stats = checkpoint['normalization_stats']
            cat_to_code = checkpoint['cat_to_code']
            model_state = checkpoint['model_state_dict']
            num_classes = checkpoint['num_classes']
            vocab_size = checkpoint['vocab_size']
            model = TransactionClassifier(
                num_classes=num_classes,
                d_model=128, 
                nhead=4, 
                num_layers=2, 
                vocab_size=vocab_size, 
                num_numerical_features=10 
            ).to(device)
            
            model.load_state_dict(model_state)
            model.eval()
            X_test_full = df.loc[X_test.index].copy()
            X_test_full['category_code'] = y_test.map(cat_to_code).fillna(-1).astype(int)
    
            test_dataset = TransactionDataset(
                X_test_full, 
                vocab=vocab_map, 
                normalization_stats=normalization_stats
            )
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            code_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    text = batch['text'].to(device)
                    nums = batch['numericals'].to(device)
                    
                    outputs = model(text, nums)
                    _, predicted = torch.max(outputs, 1)
                    code_preds.extend(predicted.cpu().numpy())
            
            code_to_cat = {v: k for k, v in cat_to_code.items()}
            preds = [code_to_cat[p] for p in code_preds]

             
    # Evaluate Model
    print('Evaluating model...')
    print('Classification Report:')
    print(classification_report(y_test, preds))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))

    return model

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/models/catboost.json'
    load_model(config)