from itertools import combinations
import numpy as np
import pandas as pd

def cluster_amounts(df, amount_col='amount', tolerance=0.10):
    """
    Clusters values in df[amount_col] such that each cluster 
    contains values within ±tolerance of the cluster median.
    Returns df with a new column 'cluster_id'.
    """
    
    # Sort by amount
    temp = df.copy().sort_values(amount_col)
    amounts = temp[amount_col].values

    clusters = []
    current_cluster = [amounts[0]]

    for value in amounts[1:]:
        median_val = np.median(current_cluster)

        # Check if within tolerance range
        if abs(value - median_val) <= tolerance * median_val:
            current_cluster.append(value)
        else:
            # Close this cluster, start a new one
            clusters.append(current_cluster)
            current_cluster = [value]

    # Append last cluster
    clusters.append(current_cluster)

    # Assign cluster IDs back to original dataframe
    cluster_id_map = {}
    cluster_id = 0

    for cluster in clusters:
        for val in cluster:
            cluster_id_map.setdefault(val, [])
            cluster_id_map[val].append(cluster_id)
        cluster_id += 1

    # If multiple equal amounts appear, assign the cluster in sequential order
    cluster_ids = []
    counter = {val: 0 for val in cluster_id_map.keys()}
    
    for val in df[amount_col]:
        cid_list = cluster_id_map[val]
        cid = cid_list[counter[val]]
        cluster_ids.append(cid)
        counter[val] += 1

    df = df.copy()
    df['cluster_id'] = cluster_ids
    return df



def detect_recurring_transactions(df, date_col='posted_date', cluster_col='cluster_id'):
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['is_recurring_income'] = False
    
    def matches_pattern(d1, d2):
        diff = abs((d2 - d1) / np.timedelta64(1, 'D'))
        return (
            (5 <= diff <= 9) or        # weekly
            (12 <= diff <= 16) or      # biweekly
            (28 <= diff <= 33)         # monthly
        )
    
    for cid, temp in df.groupby(cluster_col):
        
        temp = temp.sort_values(date_col)
        dates = temp[date_col].values
        idxs  = temp.index.values
        
        n = len(dates)
        if n < 3:
            continue
        
        recurring_groups = []

        for i in range(n):
            group_dates = [dates[i]]
            group_idxs  = [idxs[i]]

            j = i + 1

            # OPTIMIZED: only compare within ~1 month of the LAST matched date
            while j < n and (dates[j] - group_dates[-1]) <= np.timedelta64(33, 'D'):

                if matches_pattern(group_dates[-1], dates[j]):
                    group_dates.append(dates[j])
                    group_idxs.append(idxs[j])

                j += 1

            # keep groups with >= 3 transactions
            if len(group_dates) >= 3:
                recurring_groups.append(group_idxs)

        # flatten unique transaction indices
        recurring_idxs = {idx for grp in recurring_groups for idx in grp}

        df.loc[list(recurring_idxs), 'is_recurring_income'] = True
    
    return df