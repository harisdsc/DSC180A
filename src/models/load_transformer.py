from torch import torch
import torch.utils.data as data

from src.models.transformer import TransactionClassifier, TransactionDataset

def load_transformer(model_file, df, X_test, y_test):
    checkpoint = torch.load(model_file, weights_only=True)
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
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
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

    return preds