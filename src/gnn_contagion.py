# src/gnn_contagion.py
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

# Try importing Node2Vec, provide fallback
try:
    from karateclub import Node2Vec
    HAS_KARATECLUB = True
except ImportError:
    logger.warning("KarateClub not installed. GNN features will be disabled.")
    HAS_KARATECLUB = False

class GraphEmbedder:
    """
    Graph embedding and contagion risk computation.
    
    Precomputes embeddings on training graph and provides
    fallback for unseen nodes in val/test.
    """
    
    def __init__(
        self,
        embed_dim: int = 12,
        walk_length: int = 10,
        walk_number: int = 20,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4
    ):
        self.embed_dim = embed_dim
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.p = p
        self.q = q
        self.workers = workers
        
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.merchant_embeddings: Dict[str, np.ndarray] = {}
        self.user_contagion: Dict[str, float] = {}
        self.graph = None
        
    def fit(self, train: pd.DataFrame) -> 'GraphEmbedder':
        """
        Build bipartite graph from training data and compute embeddings.
        
        Args:
            train: Training DataFrame with 'user_id', 'merchant_id', 'Class'
            
        Returns:
            self
        """
        if not HAS_KARATECLUB:
            logger.warning("Skipping GNN - KarateClub not available")
            self._fallback_random_embeddings(train)
            self._compute_contagion_risk(train)
            return self
        
        logger.info("Building bipartite user-merchant graph...")
        
        # Build graph with string keys for consistency
        self.graph = nx.Graph()
        
        for _, row in tqdm(train.iterrows(), total=len(train), desc="Building graph"):
            user = f"u_{str(row['user_id'])}"
            merchant = f"m_{str(row['merchant_id'])}"
            
            if self.graph.has_edge(user, merchant):
                self.graph[user][merchant]['weight'] += 1
            else:
                self.graph.add_edge(user, merchant, weight=1)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges")
        
        # Remove isolates
        isolated = list(nx.isolates(self.graph))
        if isolated:
            logger.warning(f"Found {len(isolated)} isolated nodes, removing...")
            self.graph.remove_nodes_from(isolated)
        
        # Compute embeddings
        logger.info("Computing Node2Vec embeddings...")
        
        try:
            model = Node2Vec(
                walk_length=self.walk_length,
                walk_number=self.walk_number,
                dimensions=self.embed_dim,
                p=self.p,
                q=self.q,
                workers=self.workers
            )
            
            model.fit(self.graph)
            
            nodes = list(self.graph.nodes())
            embeddings = model.get_embedding()
            
            for idx, node in tqdm(enumerate(nodes), total=len(nodes), desc="Extracting embeddings"):
                embedding = embeddings[idx]
                
                if node.startswith('u_'):
                    user_id = node[2:]
                    self.user_embeddings[user_id] = embedding
                elif node.startswith('m_'):
                    merchant_id = node[2:]
                    self.merchant_embeddings[merchant_id] = embedding
            
            logger.info(f"Embeddings computed: {len(self.user_embeddings)} users, "
                        f"{len(self.merchant_embeddings)} merchants")
            
        except Exception as e:
            logger.error(f"Node2Vec failed: {e}")
            logger.info("Using random embeddings as fallback")
            self._fallback_random_embeddings(train)
        
        # Compute contagion risk
        self._compute_contagion_risk(train)
        
        return self
    
    def _fallback_random_embeddings(self, train: pd.DataFrame):
        """Create random embeddings if Node2Vec fails."""
        np.random.seed(42)
        
        unique_users = train['user_id'].astype(str).unique()
        unique_merchants = train['merchant_id'].astype(str).unique()
        
        for user_id in unique_users:
            self.user_embeddings[user_id] = np.random.randn(self.embed_dim) * 0.1
        
        for merchant_id in unique_merchants:
            self.merchant_embeddings[merchant_id] = np.random.randn(self.embed_dim) * 0.1
    
    def _compute_contagion_risk(self, train: pd.DataFrame):
        """
        Compute contagion risk: average fraud rate of connected merchants (train only).
        """
        logger.info("Computing contagion risk scores...")
        
        # Merchant fraud rate (train only)
        merchant_fraud = train.groupby('merchant_id')['Class'].agg(['sum', 'count'])
        merchant_fraud['fraud_rate'] = merchant_fraud['sum'] / merchant_fraud['count']
        merchant_fraud_dict = merchant_fraud['fraud_rate'].to_dict()
        
        # User to merchants mapping
        user_merchants = train.groupby('user_id')['merchant_id'].apply(set).to_dict()
        
        for user_id in tqdm(user_merchants.keys(), desc="Computing contagion"):
            merchants = user_merchants[user_id]
            fraud_rates = [merchant_fraud_dict.get(m, 0.0) for m in merchants]
            self.user_contagion[str(user_id)] = np.mean(fraud_rates) if fraud_rates else 0.0
        
        logger.info(f"Contagion risk computed for {len(self.user_contagion)} users")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add embedding and contagion features to DataFrame.
        
        Handles unseen users with zero vectors.
        """
        logger.info(f"Adding GNN features to {len(df)} transactions...")
        
        zero_embed = np.zeros(self.embed_dim)
        zero_contagion = 0.0
        
        # User embeddings (string keys)
        user_ids_str = df['user_id'].astype(str)
        user_embed_matrix = np.array([
            self.user_embeddings.get(uid, zero_embed) for uid in user_ids_str
        ])
        
        for i in range(self.embed_dim):
            df[f'user_embed_{i}'] = user_embed_matrix[:, i]
        
        df['contagion_risk'] = user_ids_str.map(self.user_contagion).fillna(zero_contagion)
        
        known_users = user_ids_str.isin(self.user_embeddings.keys()).sum()
        logger.info(f"GNN features added: {known_users}/{len(df)} known users")
        
        return df
    
    def save(self, path: str):
        """Save embedder state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'user_embeddings': self.user_embeddings,
            'merchant_embeddings': self.merchant_embeddings,
            'user_contagion': self.user_contagion,
            'embed_dim': self.embed_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Graph embedder saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GraphEmbedder':
        """Load embedder from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        embedder = cls(embed_dim=state['embed_dim'])
        embedder.user_embeddings = state['user_embeddings']
        embedder.merchant_embeddings = state['merchant_embeddings']
        embedder.user_contagion = state['user_contagion']
        
        logger.info(f"Graph embedder loaded from {path}")
        return embedder

def build_and_embed_graph(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    embed_dim: int = 12,
    cache_path: str = 'models/graph_embedder.pkl'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, GraphEmbedder]:
    """
    Fit or load GraphEmbedder and transform all splits.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached graph embedder from {cache_path}")
        embedder = GraphEmbedder.load(cache_path)
    else:
        embedder = GraphEmbedder(embed_dim=embed_dim)
        embedder.fit(train)
        embedder.save(cache_path)
    
    train = embedder.transform(train.copy())
    val = embedder.transform(val.copy())
    test = embedder.transform(test.copy())
    
    return train, val, test, embedder

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # For standalone testing
    from load_data import load_and_split
    from feature_engineering import engineer_features
    
    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    train, val, test, embedder = build_and_embed_graph(train, val, test)
    
    print("\nGNN Features Added:")
    gnn_cols = [c for c in train.columns if 'embed' in c or 'contagion' in c]
    print(train[gnn_cols].head())
    print("\nContagion risk stats:")
    print(train['contagion_risk'].describe())