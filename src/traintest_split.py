import pandas as pd
from sklearn.model_selection import train_test_split

def train_test():
    inflow = pd.read_parquet('/uss/hdsi-prismdata/q1-ucsd-inflows.pqt')
    outflow = pd.read_parquet('/uss/hdsi-prismdata/q1-ucsd-outflows.pqt')
    
    # remove rows where memo = category
    outflow = outflow[outflow['memo'] != outflow['category']]
    
    outflow_ids = set(outflow["prism_consumer_id"].unique())
    inflow_ids = set(inflow["prism_consumer_id"].unique())
    
    in_not_out = inflow_ids - outflow_ids
    out_not_in = outflow_ids - inflow_ids
    
    #consumers in both inflow and outflow
    consumers_both = sorted(set(inflow["prism_consumer_id"]).intersection(outflow["prism_consumer_id"]))
    
    #80-20 train test split
    train_ids, test_ids = train_test_split(consumers_both, test_size=0.2, random_state=42)
    
    inflow_train = inflow[inflow["prism_consumer_id"].isin(train_ids)]
    inflow_test  = inflow[inflow["prism_consumer_id"].isin(test_ids)]
    
    outflow_train = outflow[outflow["prism_consumer_id"].isin(train_ids)]
    outflow_test  = outflow[outflow["prism_consumer_id"].isin(test_ids)]
    
    return inflow_train, inflow_test, outflow_train, outflow_test