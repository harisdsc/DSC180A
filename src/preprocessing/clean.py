from src.preprocessing.rules import TransactionCleaner
import sys
import json
import time
import os
import pandas as pd

def clean_memo(config):
    print("Starting preprocessing...")
    with open(config) as f:
        config = json.load(f)
    output_file = config['output_memos']

    print(f"Loading data...")

    data = '/uss/hdsi-prismdata/q1-ucsd-outflows.pqt' if os.path.exists('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt') \
        else 'data/outflows.pqt'
    df = pd.read_parquet(data)
    df = df[df['memo'] != df['category']]

    cleaner = TransactionCleaner()

    print("Processing transactions...")
    clean_start = time.time()
    df['clean_memo'] = df['memo'].apply(cleaner.clean)
    print(f"Cleaning completed in {time.time() - clean_start:.2f} seconds")

    print(f"Saving cleaned data to {output_file}...")
    df['clean_memo'].to_csv(output_file, index=False)

if __name__ == '__main__':
    args = sys.argv
    config = args[1] if len(args) > 1 else 'configs/preprocessing/clean.json'
    clean_memo(config)
