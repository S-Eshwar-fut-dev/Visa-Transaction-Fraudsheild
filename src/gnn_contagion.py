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
        
        self.user_embeddings = {}
        self.merchant_embeddings = {}
        self.user_contagion = {}
        self.graph = None
        
    def fit(self, train: pd.DataFrame) -> 'GraphEmbedder':
        """
        Build bipartite graph from training data and compute embeddings.
        
        Args:
            train: Training DataFrame
            
        Returns:
            self
        """
        if not HAS_KARATECLUB:
            logger.warning("Skipping GNN - KarateClub not available")
            return self
        
        logger.info("Building bipartite user-merchant graph...")
        
        # Build graph
        self.graph = nx.Graph()
        
        # Add edges with progress bar
        for _, row in tqdm(train.iterrows(), total=len(train), desc="Building graph"):
            user = f"u_{row['user_id']}"
            merchant = f"m_{row['merchant_id']}"
            
            if self.graph.has_edge(user, merchant):
                self.graph[user][merchant]['weight'] += 1
            else:
                self.graph.add_edge(user, merchant, weight=1)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        # Check for isolated nodes
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
            for node in tqdm(self.graph.nodes(), desc="Extracting embeddings"):
                try:
                    embedding = model.get_embedding()[list(self.graph.nodes()).index(node)]
                    
                    if node.startswith('u_'):
                        user_id = node[2:]  # Remove 'u_' prefix
                        self.user_embeddings[user_id] = embedding
                    elif node.startswith('m_'):
                        merchant_id = node[2:]  # Remove 'm_' prefix
                        self.merchant_embeddings[merchant_id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to get embedding for {node}: {e}")
            
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
        
        for user_id in train['user_id'].unique():
            self.user_embeddings[user_id] = np.random.randn(self.embed_dim) * 0.1
        
        for merchant_id in train['merchant_id'].unique():
            self.merchant_embeddings[merchant_id] = np.random.randn(self.embed_dim) * 0.1
    
    def _compute_contagion_risk(self, train: pd.DataFrame):
        """
        Compute contagion risk: average fraud rate of connected merchants.
        
        This captures fraud ring behavior - users connected to high-fraud
        merchants are at higher risk.
        """
        logger.info("Computing contagion risk scores...")
        
        # Calculate fraud rate per merchant (train only)
        merchant_fraud = train.groupby('merchant_id')['Class'].agg(['sum', 'count']).reset_index()
        merchant_fraud['fraud_rate'] = merchant_fraud['sum'] / merchant_fraud['count']
        merchant_fraud_dict = dict(zip(merchant_fraud['merchant_id'], merchant_fraud['fraud_rate']))
        
        # For each user, compute average fraud rate of their merchants
        user_merchants = train.groupby('user_id')['merchant_id'].apply(set).to_dict()
        
        for user_id, merchants in tqdm(user_merchants.items(), desc="Computing contagion"):
            fraud_rates = [merchant_fraud_dict.get(m, 0) for m in merchants]
            self.user_contagion[user_id] = np.mean(fraud_rates) if fraud_rates else 0.0
        
        logger.info(f"Contagion risk computed for {len(self.user_contagion)} users")
        logger.info(f"Mean contagion: {np.mean(list(self.user_contagion.values())):.4f}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add embedding and contagion features to DataFrame.
        
        Handles unseen users/merchants with zero vectors.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with GNN features
        """
        logger.info(f"Adding GNN features to {len(df)} transactions...")
        
        # Default zero vector for unseen nodes
        zero_user_embed = np.zeros(self.embed_dim)
        zero_contagion = 0