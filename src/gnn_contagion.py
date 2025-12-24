import networkx as nx
import pandas as pd
import json
from karateclub import Node2Vec
from typing import Dict

def build_and_embed_graph(train: pd.DataFrame, embed_dim: int = 8) -> pd.DataFrame:
    """Bipartite user-merchant graph + Node2Vec embeds + contagion risk."""
    G = nx.Graph()
    for _, row in train.iterrows():
        G.add_edge(row['user_id'], row['merchant_id'], weight=1.0)
    
    # Node2Vec (fast params for prep)
    n2v = Node2Vec(walk_length=10, walk_number=20, dimensions=embed_dim)
    n2v.fit(G)
    
    embeds = {node: n2v.get_embedding(node) for node in G.nodes}
    
    # User embeds DF
    user_embeds = {k: v for k, v in embeds.items() if k in train['user_id'].unique()}
    embed_df = pd.DataFrame.from_dict(user_embeds, orient='index')
    embed_df['user_id'] = list(user_embeds.keys())
    embed_cols = [f'user_embed_{i}' for i in range(embed_dim)]
    embed_df = embed_df[['user_id'] + embed_cols].fillna(0)
    
    # Contagion: Avg neighbor fraud rate (train-only)
    fraud_rates = {}
    for uid in train['user_id'].unique():
        neighs = list(G.neighbors(uid))
        if neighs:
            neigh_fraud = train[train['merchant_id'].isin(neighs)]['Class'].mean()
            fraud_rates[uid] = neigh_fraud
        else:
            fraud_rates[uid] = 0
    
    # Merge to train
    train = train.merge(embed_df, on='user_id', how='left').fillna(0)
    train['contagion_risk'] = train['user_id'].map(fraud_rates).fillna(0)
    
    # Export for cache/API
    export = {'embeds': embeds, 'rates': fraud_rates}
    with open('models/precomputed_embeds.json', 'w') as f:
        json.dump(export, f)
    
    return train

if __name__ == "__main__":
    from load_data import load_and_split
    from feature_engineering import engineer_features
    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    train = build_and_embed_graph(train)
    print("GNN added: Contagion mean", train['contagion_risk'].mean())