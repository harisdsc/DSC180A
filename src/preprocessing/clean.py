import rules
import sys
import json
import time
import pandas as pd

def clean_memo(config_path):
    # 1. Load Configuration
    print("Starting preprocessing...")
    with open(config_path) as f:
        config = json.load(f)

    output_file = config['output_memos']

    # 2. Load Data
    print(f"Loading data...")
    df = pd.read_parquet('data/outflows.pqt')

    # Initialize Cleaner
    cleaner = rules.TransactionCleaner()

    # 3. Apply Cleaning Logic
    print("Processing transactions...")
    df['clean_memo'] = df['memo'].apply(cleaner.clean)

    # 4. Save Output
    print(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

    return df

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/preprocessing/clean.json'
    clean_memo(config)
