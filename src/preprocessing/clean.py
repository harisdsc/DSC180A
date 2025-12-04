from src.preprocessing.rules import TransactionCleaner
import sys
import json
import pandas as pd

def clean_memo(config):
    # 1. Load Configuration
    print("Starting preprocessing...")
    with open(config) as f:
        config = json.load(f)

    output_file = config['output_memos']

    # 2. Load Data
    print(f"Loading data...")
    df = pd.read_parquet('data/outflows.pqt')
    df = df[df['memo'] != df['category']]

    # Initialize Cleaner
    cleaner = TransactionCleaner()

    # 3. Apply Cleaning Logic
    print("Processing transactions...")
    df['clean_memo'] = df['memo'].apply(cleaner.clean)

    # 4. Save Output
    print(f"Saving cleaned data to {output_file}...")
    df['clean_memo'].to_csv(output_file, index=False)
    print("Done.")

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/preprocessing/clean.json'
    clean_memo(config)
