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
        if n < 3: #<-- uncomment section
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
            if len(group_dates) >= 3:  #<-- uncomment
                recurring_groups.append(group_idxs)

        # flatten unique transaction indices
        recurring_idxs = {idx for grp in recurring_groups for idx in grp}

        df.loc[list(recurring_idxs), 'is_recurring_income'] = True
    
    return df


def k_median_1d(amounts, K):
    """
    Run 1D K-median clustering with median center updates.
    """
    amounts = np.sort(np.array(amounts))

    # initialization: evenly spaced centers
    centers = np.linspace(amounts.min(), amounts.max(), K)

    while True:
        # Assign points to nearest center
        clusters = [[] for _ in range(K)]
        for x in amounts:
            idx = np.argmin(np.abs(centers - x))
            clusters[idx].append(x)

        # Recompute medians
        new_centers = np.array(
            [np.median(cluster) if len(cluster) > 0 else centers[i]
             for i, cluster in enumerate(clusters)]
        )

        # Converged?
        if np.max(np.abs(new_centers - centers)) < 1e-6:
            break

        centers = new_centers

    return clusters, centers



def silhouette_1d(amounts, clusters):
    """
    1D silhouette score for K-median clustering.
    """
    amounts = np.sort(np.array(amounts))
    K = len(clusters)

    # Build index -> cluster_id mapping
    labels = np.zeros(len(amounts), dtype=int)
    idx = 0
    for k, cluster in enumerate(clusters):
        for _ in cluster:
            labels[idx] = k
            idx += 1

    scores = []

    for i, x in enumerate(amounts):
        k = labels[i]
        own_cluster = np.array(clusters[k])

        # a(i) = mean distance to own cluster
        a = np.mean(np.abs(own_cluster - x))

        # b(i) = smallest mean distance to another cluster
        b = np.inf
        for j in range(K):
            if j == k or len(clusters[j]) == 0:
                continue
            b = min(b, np.mean(np.abs(np.array(clusters[j]) - x)))

        s = (b - a) / max(a, b)
        scores.append(s)

    return np.mean(scores)


def find_best_k_for_consumer(amounts, kmax=10, tol=0.10):
    amounts = np.array(amounts)

    if len(amounts) <= 2:
        return 1, [[x for x in amounts]], np.array([np.median(amounts)]), 0

    best_score = -np.inf
    best_clusters = None
    best_centers = None
    best_k = 1

    for K in range(2, min(kmax, len(amounts)) + 1):
        clusters, centers = k_median_1d(amounts, K)
        clusters = [c for c in clusters if len(c) > 0]
        centers = np.array([np.median(c) for c in clusters])

        # split clusters if they violate tolerance -----
        refined_clusters = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            refined_clusters.extend(split_cluster_by_tolerance(cluster, tol=tol))
            
        if len(refined_clusters) == 0:
            # fallback: everything is one cluster
            refined_clusters = [list(amounts)]

        # compute centers of refined clusters
        refined_centers = np.array([np.median(c) for c in refined_clusters])

        # silhouette score for refined clustering
        score = silhouette_1d(amounts, refined_clusters)

        if np.isnan(score):
            score = -np.inf
    
        if score > best_score:
            best_score = score
            best_clusters = refined_clusters
            best_centers = refined_centers
            best_k = len(refined_clusters)

        if best_clusters is None:
            best_clusters = [list(amounts)]
            best_centers = [np.median(amounts)]
            best_k = 1

    return best_k, best_clusters, best_centers, best_score


def assign_cluster_ids(amounts, clusters):
    """
    Convert clusters (list of lists of amounts) back into a cluster_id array.
    """
    clusters = [c for c in clusters if len(c) > 0]  # remove empties

    if len(clusters) == 0:                          # FALLBACK
        return np.zeros(len(amounts), dtype=int)

    amounts_sorted = np.sort(np.array(amounts))
    labels_sorted = np.zeros(len(amounts_sorted), dtype=int)

    idx = 0
    for cid, cluster in enumerate(clusters):
        for _ in cluster:
            labels_sorted[idx] = cid
            idx += 1

    # Map back to original order
    mapping = dict(zip(amounts_sorted, labels_sorted))
    return np.array([mapping[x] for x in amounts])


def split_cluster_by_tolerance(cluster, tol=0.10):
    """
    Given a single cluster (list of values), split it into subclusters
    where each subcluster satisfies ±tol of its own median.
    """
    cluster = np.sort(np.array(cluster))
    if len(cluster) == 0:
        return []

    subclusters = []

    current = [cluster[0]]

    for x in cluster[1:]:
        med = np.median(current)
        if abs(x - med) <= tol * med:
            current.append(x)
        else:
            subclusters.append(current)
            current = [x]

    subclusters.append(current)
    return subclusters

