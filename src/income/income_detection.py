import pandas as pd
from joblib import Parallel, delayed
import time
# import sys
# import os
# repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
# sys.path.append(repo_root)

from income_funcs import (cluster_amounts, detect_recurring_transactions, k_median_1d, 
silhouette_1d, find_best_k_for_consumer, assign_cluster_ids, split_cluster_by_tolerance
)

# from src.traintest_split import train_test inflow_train, inflow_test, outflow_train, outflow_test = train_test()

# Load dataframe (NO sorting yet)
df = pd.read_parquet('/uss/hdsi-prismdata/q1-ucsd-inflows.pqt')
df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')

# sort dates per consumer
it_sorted = df.sort_values(['prism_consumer_id', 'posted_date'])
it_sorted['is_recurring_income'] = False

# Process one consumer
def process_one_consumer(cid, df_sub):
    df_sub = df_sub.copy()
    amts = df_sub['amount'].values
    best_k, clusters, centers, score = find_best_k_for_consumer(amts)
    df_sub['cluster_id'] = assign_cluster_ids(amts, clusters)
    df_sub = detect_recurring_transactions(df_sub)
    return df_sub


# Run parallel processing (fast version)
def parallel_process_by_consumer(df, n_jobs=-1, verbose=True):

    df_grouped = df.groupby(["prism_consumer_id"])
    groups = list(df_grouped)

    if verbose:
        print(f"Starting parallel processing for {len(groups)} consumers...")

    start = time.time()

    results = Parallel(n_jobs=n_jobs)(
    delayed(process_one_consumer)(key, g)
    for key, g in df_grouped
)


    end = time.time()
    total_latency = end - start

    if verbose:
        print("\n================ LATENCY REPORT ================")
        print(f"Total consumers processed: {len(groups)}")
        print(f"Wall-clock latency: {total_latency:.4f} seconds")
        print(f"Latency per consumer: {total_latency/len(groups):.8f} seconds")
        print("================================================\n")

    return pd.concat(results).sort_index()

income = parallel_process_by_consumer(it_sorted, n_jobs=-1)

# Flag income categories
income_cats = [
    'PAYCHECK', 'PAYCHECK_PLACEHOLDER', 'INVESTMENT_INCOME',
    'OTHER_BENEFITS', 'UNEMPLOYMENT_BENEFITS', 'TAX'
]

income.loc[income['category'].isin(income_cats), 'is_recurring_income'] = True