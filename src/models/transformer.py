import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import time
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

def build_vocab(text_list, max_vocab=20002):
    words = [str(t).lower().split() for t in text_list]
    all_words = [item for sublist in words for item in sublist]
    counts = Counter(all_words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(text_list, vocab, max_len=50):
    tokenized_batch = []
    for text in text_list:
        words = str(text).lower().split()
        indices = [vocab.get(w, 1) for w in words[:max_len]]
        tokenized_batch.append(torch.tensor(indices, dtype=torch.long))
    padded = pad_sequence(tokenized_batch, batch_first=True, padding_value=0)
    if padded.size(1) < max_len:
        zeros = torch.zeros(padded.size(0), max_len - padded.size(1), dtype=torch.long)
        padded = torch.cat([padded, zeros], dim=1)
    else:
        padded = padded[:, :max_len]
    return padded

# --- Dataset ---
class TransactionDataset(data.Dataset):
    # Added normalization_stats parameter
    def __init__(self, df, vocab, max_seq_len=50, normalization_stats=None):
        self.memos = df["clean_memo"].tolist()
        df = df.copy() # Avoid SettingWithCopy warnings
        df['whole_dollar'] = df['whole_dollar'].astype(float)

        cols_to_normalize = [
            'days_until_next_holiday', 
            'days_since_prev_holiday', 
            'month_med_amnt_diff', 
            'rolling_avg_days_between_txn',
            'days_since_prev'
        ]

        # Logic to handle Training vs Inference normalization
        self.normalization_stats = {}
        
        for col in cols_to_normalize:
            df[col] = df[col].fillna(0)
            
            if normalization_stats is None:
                # TRAINING MODE: Calculate and store stats
                mean = df[col].mean()
                std = df[col].std()
                self.normalization_stats[col] = {'mean': mean, 'std': std}
            else:
                # INFERENCE MODE: Use loaded stats
                mean = normalization_stats[col]['mean']
                std = normalization_stats[col]['std']

            # Apply normalization
            df[f"{col}_z"] = (df[col] - mean) / (std + 1e-6)

        # ... (The rest of your numericals setup remains exactly the same) ...
        extra_cols = [
            "dow_sin", "dow_cos", 
            "month_sin", "month_cos", 
            "log_amnt",                       
            "whole_dollar",                 
            "days_until_next_holiday_z",    
            "days_since_prev_holiday_z",    
            "month_med_amnt_diff_z",        
            "rolling_avg_days_between_txn_z"
        ]
        clean_numericals = df[extra_cols].fillna(0.0).values.astype(np.float32)
        clean_numericals = np.nan_to_num(clean_numericals, nan=0.0, posinf=0.0, neginf=0.0)
        self.numericals = clean_numericals
        
        if "category_code" in df.columns:
            self.labels = df["category_code"].values.astype(np.int64)
        else:
            self.labels = np.zeros(len(df), dtype=np.int64)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.text_tokens = tokenize_and_pad(self.memos, self.vocab, self.max_seq_len)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text_tokens[idx],
            'numericals': torch.tensor(self.numericals[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ... [Keep PositionalEncoding and TransactionClassifier exactly as they were] ...
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransactionClassifier(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=2, vocab_size=20002, num_numerical_features=8, dropout=0.1):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        hidden_dim = d_model * 2
        self.num_projection = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, text_input, numerical_features):
        batch_size = text_input.size(0)
        text_emb = self.text_embedding(text_input) 
        text_emb = self.pos_encoder(text_emb)
        num_emb = self.num_projection(numerical_features).unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        combined_sequence = torch.cat([cls_tokens, text_emb, num_emb], dim=1)
        transformer_out = self.transformer_encoder(combined_sequence)
        
        cls_output = transformer_out[:, 0, :] 
        logits = self.classifier(cls_output)
        
        return logits

def train_transformer(X_train, y_train, X_test, y_test, model_file):
    train_start = time.time()
    print("Building vocabulary...")
    df = X_train.copy()
    df["category"] = y_train
    
    vocab_map = build_vocab(df["clean_memo"].tolist())
    vocab_size = len(vocab_map)
    train_categories = df["category"].astype("category").cat.categories
    cat_to_code = {cat: i for i, cat in enumerate(train_categories)}
    
    num_classes = len(cat_to_code)
    
    df["category_code"] = df["category"].map(cat_to_code).fillna(-1).astype(int)
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    
    print(f"Initializing dataset on {DEVICE}. Num classes: {num_classes}")
    
    train_dataset = TransactionDataset(df, vocab_map)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TransactionClassifier(
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        vocab_size=vocab_size,
        num_numerical_features=10,
        dropout=0.1
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training Transformer...")
    total_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            text = batch['text'].to(DEVICE)
            nums = batch['numericals'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(text, nums)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # ### FIX 2: Gradient Clipping ###
            # Transformers are unstable. If gradients get too large, they explode to NaN.
            # This limits the "norm" (size) of the gradients to 1.0 before we update weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 4000 == 0 and batch_idx > 0:
                print(f"Step [{batch_idx}], Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"--- Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}% ---")
        print(f"Epoch {epoch+1} completed in {time.time() - epoch_start_time:.2f} seconds.")

    print(f"Training completed in {time.time() - total_start_time:.2f} seconds.")
    
    print(f"Saving model to {model_file}...")
    
    save_content = {
        'model_state_dict': model.state_dict(),
        'vocab_map': vocab_map,
        'cat_to_code': cat_to_code,
        'num_classes': num_classes,
        'vocab_size': vocab_size,
        # ADD THIS LINE:
        'normalization_stats': train_dataset.normalization_stats 
    }
    torch.save(save_content, model_file)
    print("Model saved successfully.")

    # Test Loop
    df_test = X_test.copy()
    df_test["category"] = y_test
    df_test["category_code"] = df_test["category"].map(cat_to_code).fillna(-1).astype(int)
    
    test_dataset = TransactionDataset(df_test, vocab_map)
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    all_preds = []
    
    print("Predicting...")
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(DEVICE)
            nums = batch['numericals'].to(DEVICE)
            outputs = model(text, nums)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    code_to_cat = {v: k for k, v in cat_to_code.items()}
    all_preds_strings = [code_to_cat[p] for p in all_preds]
    
    return all_preds_strings