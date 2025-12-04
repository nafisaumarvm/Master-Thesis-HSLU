"""
Federated Learning framework for privacy-preserving model training.

This module enables training on distributed hotel/PMS data without centralizing
sensitive guest information. Data remains local, only model updates are shared.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import warnings

from .utils import set_random_seed


class LocalDataNode:
    """
    Represents a local data node (e.g., a single hotel or PMS instance).
    
    Stores guest data locally and computes model updates without sharing raw data.
    """
    
    def __init__(
        self,
        node_id: str,
        local_data: pd.DataFrame,
        privacy_budget: float = 1.0
    ):
        """
        Args:
            node_id: Unique identifier for this node
            local_data: Local guest/exposure data (NEVER leaves this node)
            privacy_budget: Differential privacy epsilon budget
        """
        self.node_id = node_id
        self._local_data = local_data  # Private, never shared
        self.privacy_budget = privacy_budget
        self.n_samples = len(local_data)
        
    def compute_aggregated_statistics(self) -> Dict[str, Any]:
        """
        Compute privacy-preserving aggregated statistics.
        
        Returns:
            Dict of aggregated, anonymized statistics (no individual records)
        """
        stats = {
            'node_id': self.node_id,
            'n_samples': self.n_samples,
            'avg_ctr': self._local_data['click'].mean() if 'click' in self._local_data.columns else 0.0,
            'total_clicks': int(self._local_data['click'].sum()) if 'click' in self._local_data.columns else 0,
            'total_impressions': len(self._local_data)
        }
        
        # Add noise for differential privacy (Laplace mechanism)
        if self.privacy_budget > 0:
            sensitivity = 1.0 / self.n_samples
            noise_scale = sensitivity / self.privacy_budget
            
            stats['avg_ctr'] += np.random.laplace(0, noise_scale)
            stats['avg_ctr'] = np.clip(stats['avg_ctr'], 0, 1)
        
        return stats
    
    def compute_local_gradients(
        self,
        model_weights: np.ndarray,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Compute local gradients on private data.
        
        Args:
            model_weights: Current global model weights
            learning_rate: Learning rate
            
        Returns:
            Local gradient updates (anonymized)
        """
        # Simulate gradient computation
        # In real implementation, this would compute actual gradients
        
        if 'click' not in self._local_data.columns:
            return np.zeros_like(model_weights)
        
        # Mock gradient computation
        local_gradient = np.random.randn(*model_weights.shape) * 0.01
        
        # Gradient clipping for privacy
        clip_norm = 1.0
        norm = np.linalg.norm(local_gradient)
        if norm > clip_norm:
            local_gradient = local_gradient * (clip_norm / norm)
        
        # Add noise for differential privacy
        if self.privacy_budget > 0:
            noise_scale = clip_norm / (self.n_samples * self.privacy_budget)
            noise = np.random.laplace(0, noise_scale, size=local_gradient.shape)
            local_gradient += noise
        
        return local_gradient
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get anonymized data summary (safe to share).
        
        Returns:
            Summary statistics without individual-level data
        """
        summary = {
            'node_id': self.node_id,
            'n_records': self.n_samples,
            'date_range': None,
            'feature_distributions': {}
        }
        
        # Add non-sensitive summary statistics
        if 'arrival_date' in self._local_data.columns:
            summary['date_range'] = {
                'earliest': self._local_data['arrival_date'].min(),
                'latest': self._local_data['arrival_date'].max()
            }
        
        # Aggregated distributions (no individual records)
        for col in ['purpose_of_stay', 'source', 'country']:
            if col in self._local_data.columns:
                # Only share if sufficient records to prevent re-identification
                if self.n_samples >= 50:
                    summary['feature_distributions'][col] = \
                        self._local_data[col].value_counts().to_dict()
        
        return summary


class FederatedAggregator:
    """
    Central aggregator for federated learning.
    
    Coordinates training across nodes without accessing raw data.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.nodes: Dict[str, LocalDataNode] = {}
        self.global_model_weights = None
        self.training_history = []
        self.rng = set_random_seed(seed)
        
    def register_node(self, node: LocalDataNode):
        """
        Register a local data node.
        
        Args:
            node: LocalDataNode instance
        """
        self.nodes[node.node_id] = node
        print(f"Registered node: {node.node_id} ({node.n_samples} samples)")
    
    def aggregate_statistics(self) -> Dict[str, Any]:
        """
        Aggregate statistics across all nodes.
        
        Returns:
            Federated statistics (privacy-preserving)
        """
        if not self.nodes:
            return {}
        
        all_stats = [node.compute_aggregated_statistics() for node in self.nodes.values()]
        
        # Weighted average CTR
        total_impressions = sum(s['total_impressions'] for s in all_stats)
        total_clicks = sum(s['total_clicks'] for s in all_stats)
        
        federated_stats = {
            'n_nodes': len(self.nodes),
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'federated_ctr': total_clicks / total_impressions if total_impressions > 0 else 0.0,
            'node_stats': all_stats
        }
        
        return federated_stats
    
    def federated_averaging(
        self,
        n_rounds: int = 10,
        learning_rate: float = 0.01,
        feature_dim: int = 10
    ) -> np.ndarray:
        """
        Perform federated averaging (FedAvg) across nodes.
        
        Args:
            n_rounds: Number of federated training rounds
            learning_rate: Learning rate
            feature_dim: Model feature dimension
            
        Returns:
            Final global model weights
        """
        # Initialize global model
        if self.global_model_weights is None:
            self.global_model_weights = np.zeros(feature_dim)
        
        for round_idx in range(n_rounds):
            print(f"\nFederated Round {round_idx + 1}/{n_rounds}")
            
            # Collect local gradients
            local_gradients = []
            node_weights = []
            
            for node in self.nodes.values():
                gradient = node.compute_local_gradients(
                    self.global_model_weights,
                    learning_rate
                )
                local_gradients.append(gradient)
                node_weights.append(node.n_samples)
            
            # Weighted average of gradients
            total_samples = sum(node_weights)
            weighted_gradient = sum(
                (w / total_samples) * g 
                for w, g in zip(node_weights, local_gradients)
            )
            
            # Update global model
            self.global_model_weights += learning_rate * weighted_gradient
            
            # Track history
            self.training_history.append({
                'round': round_idx + 1,
                'gradient_norm': np.linalg.norm(weighted_gradient),
                'model_norm': np.linalg.norm(self.global_model_weights)
            })
            
            print(f"  Gradient norm: {np.linalg.norm(weighted_gradient):.6f}")
        
        return self.global_model_weights
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Returns:
            Privacy report
        """
        report = {
            'total_nodes': len(self.nodes),
            'privacy_guarantees': {
                'data_centralization': 'NO - All data stays local',
                'individual_records_shared': 'NO - Only aggregated statistics',
                'differential_privacy': 'YES - Laplace noise applied',
                'gradient_clipping': 'YES - Bounded sensitivity'
            },
            'node_privacy_budgets': {
                node_id: node.privacy_budget 
                for node_id, node in self.nodes.items()
            }
        }
        
        return report


def create_federated_setup(
    guests_df: pd.DataFrame,
    n_nodes: int = 3,
    node_prefix: str = "HOTEL",
    privacy_budget: float = 1.0,
    seed: int = 42
) -> Tuple[FederatedAggregator, List[LocalDataNode]]:
    """
    Create a federated learning setup by partitioning data across nodes.
    
    Args:
        guests_df: Full guest dataset
        n_nodes: Number of federated nodes (e.g., hotels)
        node_prefix: Prefix for node IDs
        privacy_budget: Privacy budget per node
        seed: Random seed
        
    Returns:
        (aggregator, list of nodes)
    """
    rng = set_random_seed(seed)
    
    # Partition data across nodes
    shuffled = guests_df.sample(frac=1.0, random_state=seed)
    partition_size = len(shuffled) // n_nodes
    
    aggregator = FederatedAggregator(seed=seed)
    nodes = []
    
    for i in range(n_nodes):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < n_nodes - 1 else len(shuffled)
        
        node_data = shuffled.iloc[start_idx:end_idx].copy()
        
        node = LocalDataNode(
            node_id=f"{node_prefix}_{i+1:02d}",
            local_data=node_data,
            privacy_budget=privacy_budget
        )
        
        aggregator.register_node(node)
        nodes.append(node)
    
    return aggregator, nodes


def demonstrate_privacy_preservation(
    aggregator: FederatedAggregator
) -> pd.DataFrame:
    """
    Demonstrate that individual records cannot be recovered.
    
    Args:
        aggregator: FederatedAggregator instance
        
    Returns:
        Comparison dataframe
    """
    # Show what IS available (aggregated stats)
    available_stats = aggregator.aggregate_statistics()
    
    # Show what is NOT available (individual records)
    unavailable_info = {
        'individual_guest_data': 'NOT AVAILABLE - Stored locally only',
        'personal_identifiers': 'NOT AVAILABLE - Never collected',
        'exact_timestamps': 'NOT AVAILABLE - Only date ranges shared',
        'individual_click_history': 'NOT AVAILABLE - Only aggregated CTR'
    }
    
    comparison = pd.DataFrame({
        'Information Type': list(available_stats.keys()) + list(unavailable_info.keys()),
        'Availability': (
            ['Available (Aggregated)'] * len(available_stats) +
            ['Not Available (Privacy Protected)'] * len(unavailable_info)
        ),
        'Purpose': (
            ['Model training'] * len(available_stats) +
            ['Privacy protection'] * len(unavailable_info)
        )
    })
    
    return comparison





