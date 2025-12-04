"""
Contextual bandit policies for ad selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import expit as sigmoid

from .utils import set_random_seed


class EpsilonGreedyBandit:
    """
    Epsilon-greedy contextual bandit with per-ad Beta priors.
    """
    
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        """
        Args:
            epsilon: Exploration probability
            seed: Random seed
        """
        self.epsilon = epsilon
        self.rng = set_random_seed(seed)
        
        # Beta prior parameters per ad: (alpha, beta)
        self.ad_alpha = {}  # successes + 1
        self.ad_beta = {}   # failures + 1
        
    def select_ads(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        k: int = 3
    ) -> List[str]:
        """
        Select k ads using epsilon-greedy policy.
        
        Args:
            guest_context: Guest information (unused in this simple version)
            candidate_ads: Candidate ads dataframe
            k: Number of ads to select
            
        Returns:
            List of selected ad_ids
        """
        if len(candidate_ads) == 0:
            return []
        
        # Explore vs exploit
        if self.rng.random() < self.epsilon:
            # Explore: random selection
            n_sample = min(k, len(candidate_ads))
            selected_indices = self.rng.choice(
                len(candidate_ads), size=n_sample, replace=False
            )
            return candidate_ads.iloc[selected_indices]['ad_id'].tolist()
        else:
            # Exploit: Thompson sampling from Beta posteriors
            scores = []
            
            for _, ad in candidate_ads.iterrows():
                ad_id = ad['ad_id']
                
                # Get Beta parameters (default to uniform prior)
                alpha = self.ad_alpha.get(ad_id, 1.0)
                beta = self.ad_beta.get(ad_id, 1.0)
                
                # Sample from Beta(alpha, beta)
                score = self.rng.beta(alpha, beta)
                scores.append(score)
            
            # Select top-k
            scores = np.array(scores)
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            return candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
    
    def update(self, ad_id: str, reward: float):
        """
        Update Beta parameters based on observed reward.
        
        Args:
            ad_id: Ad identifier
            reward: Binary reward (0 or 1)
        """
        if ad_id not in self.ad_alpha:
            self.ad_alpha[ad_id] = 1.0
            self.ad_beta[ad_id] = 1.0
        
        if reward > 0:
            self.ad_alpha[ad_id] += 1
        else:
            self.ad_beta[ad_id] += 1
    
    def get_ad_stats(self, ad_id: str) -> Tuple[float, float]:
        """
        Get current posterior mean and std for an ad.
        
        Args:
            ad_id: Ad identifier
            
        Returns:
            (mean, std) of Beta posterior
        """
        alpha = self.ad_alpha.get(ad_id, 1.0)
        beta = self.ad_beta.get(ad_id, 1.0)
        
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = np.sqrt(var)
        
        return mean, std


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.
    """
    
    def __init__(
        self,
        feature_dim: int,
        alpha: float = 1.0,
        seed: int = 42
    ):
        """
        Args:
            feature_dim: Dimension of context features
            alpha: Exploration parameter
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.rng = set_random_seed(seed)
        
        # Per-ad parameters
        self.A = {}  # Design matrix A_a = D^T D + I
        self.b = {}  # Response vector b_a = D^T c
        
    def _get_or_init_params(self, ad_id: str):
        """Get or initialize parameters for an ad."""
        if ad_id not in self.A:
            self.A[ad_id] = np.eye(self.feature_dim)
            self.b[ad_id] = np.zeros(self.feature_dim)
        
        return self.A[ad_id], self.b[ad_id]
    
    def select_ads(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        k: int = 3,
        context_features: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Select k ads using LinUCB.
        
        Args:
            guest_context: Guest information
            candidate_ads: Candidate ads dataframe
            k: Number of ads to select
            context_features: Pre-computed context features (optional)
            
        Returns:
            List of selected ad_ids
        """
        if len(candidate_ads) == 0:
            return []
        
        if context_features is None:
            # Build simple features from guest context
            context_features = self._build_context_features(
                guest_context, candidate_ads
            )
        
        ucb_scores = []
        
        for i, (_, ad) in enumerate(candidate_ads.iterrows()):
            ad_id = ad['ad_id']
            x = context_features[i]
            
            A_a, b_a = self._get_or_init_params(ad_id)
            
            # Compute inverse
            A_inv = np.linalg.inv(A_a)
            
            # Estimated reward
            theta_a = A_inv @ b_a
            p_t_a = theta_a @ x
            
            # Upper confidence bound
            ucb = p_t_a + self.alpha * np.sqrt(x @ A_inv @ x)
            
            ucb_scores.append(ucb)
        
        # Select top-k
        ucb_scores = np.array(ucb_scores)
        top_k_indices = np.argsort(ucb_scores)[::-1][:k]
        
        return candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
    
    def update(
        self,
        ad_id: str,
        context_feature: np.ndarray,
        reward: float
    ):
        """
        Update LinUCB parameters.
        
        Args:
            ad_id: Ad identifier
            context_feature: Context feature vector
            reward: Observed reward
        """
        A_a, b_a = self._get_or_init_params(ad_id)
        
        # Update
        self.A[ad_id] = A_a + np.outer(context_feature, context_feature)
        self.b[ad_id] = b_a + reward * context_feature
    
    def _build_context_features(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame
    ) -> np.ndarray:
        """
        Build simple context features from guest and ads.
        
        Args:
            guest_context: Guest information
            candidate_ads: Candidate ads
            
        Returns:
            Feature matrix (n_ads, feature_dim)
        """
        n_ads = len(candidate_ads)
        features = np.zeros((n_ads, self.feature_dim))
        
        # Simple feature construction
        # Feature 0: bias
        features[:, 0] = 1.0
        
        # Feature 1: distance (normalized)
        if self.feature_dim > 1 and 'distance_km' in candidate_ads.columns:
            features[:, 1] = candidate_ads['distance_km'].values / 10.0
        
        # Feature 2: base utility (normalized)
        if self.feature_dim > 2 and 'base_utility' in candidate_ads.columns:
            features[:, 2] = candidate_ads['base_utility'].values
        
        # Feature 3: is_family match
        if self.feature_dim > 3:
            is_family = guest_context.get('is_family', False)
            # Check if ad is family-friendly (simplified)
            features[:, 3] = float(is_family)
        
        # Feature 4: price match
        if self.feature_dim > 4:
            guest_price = guest_context.get('price_per_night', 100)
            # Normalized guest price
            features[:, 4] = guest_price / 200.0
        
        # Additional features can be added
        
        return features


class ThompsonSamplingLogistic:
    """
    Thompson Sampling with logistic reward model.
    """
    
    def __init__(
        self,
        feature_dim: int,
        prior_variance: float = 1.0,
        seed: int = 42
    ):
        """
        Args:
            feature_dim: Dimension of context features
            prior_variance: Prior variance for Gaussian approximation
            seed: Random seed
        """
        self.feature_dim = feature_dim
        self.prior_variance = prior_variance
        self.rng = set_random_seed(seed)
        
        # Per-ad Gaussian approximation to posterior
        self.mu = {}     # Mean
        self.Sigma = {}  # Covariance
        
    def _get_or_init_params(self, ad_id: str):
        """Get or initialize parameters for an ad."""
        if ad_id not in self.mu:
            self.mu[ad_id] = np.zeros(self.feature_dim)
            self.Sigma[ad_id] = self.prior_variance * np.eye(self.feature_dim)
        
        return self.mu[ad_id], self.Sigma[ad_id]
    
    def select_ads(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        k: int = 3,
        context_features: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Select k ads using Thompson Sampling.
        
        Args:
            guest_context: Guest information
            candidate_ads: Candidate ads dataframe
            k: Number of ads to select
            context_features: Pre-computed context features
            
        Returns:
            List of selected ad_ids
        """
        if len(candidate_ads) == 0:
            return []
        
        if context_features is None:
            context_features = self._build_context_features(
                guest_context, candidate_ads
            )
        
        sampled_scores = []
        
        for i, (_, ad) in enumerate(candidate_ads.iterrows()):
            ad_id = ad['ad_id']
            x = context_features[i]
            
            mu_a, Sigma_a = self._get_or_init_params(ad_id)
            
            # Sample theta from posterior
            theta_sample = self.rng.multivariate_normal(mu_a, Sigma_a)
            
            # Compute expected reward
            score = sigmoid(theta_sample @ x)
            sampled_scores.append(score)
        
        # Select top-k
        sampled_scores = np.array(sampled_scores)
        top_k_indices = np.argsort(sampled_scores)[::-1][:k]
        
        return candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
    
    def update(
        self,
        ad_id: str,
        context_feature: np.ndarray,
        reward: float
    ):
        """
        Update posterior using Laplace approximation.
        
        Args:
            ad_id: Ad identifier
            context_feature: Context feature vector
            reward: Observed reward (0 or 1)
        """
        mu_a, Sigma_a = self._get_or_init_params(ad_id)
        
        # Simplified update: online logistic regression approximation
        # Predict current
        p = sigmoid(mu_a @ context_feature)
        
        # Gradient
        grad = (reward - p) * context_feature
        
        # Hessian approximation (Fisher information)
        H = p * (1 - p) * np.outer(context_feature, context_feature)
        
        # Update covariance (simple online approximation)
        Sigma_inv = np.linalg.inv(Sigma_a) + H
        Sigma_new = np.linalg.inv(Sigma_inv)
        
        # Update mean
        mu_new = Sigma_new @ (np.linalg.inv(Sigma_a) @ mu_a + grad)
        
        self.mu[ad_id] = mu_new
        self.Sigma[ad_id] = Sigma_new
    
    def _build_context_features(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame
    ) -> np.ndarray:
        """Build simple context features."""
        n_ads = len(candidate_ads)
        features = np.zeros((n_ads, self.feature_dim))
        
        features[:, 0] = 1.0
        
        if self.feature_dim > 1 and 'distance_km' in candidate_ads.columns:
            features[:, 1] = candidate_ads['distance_km'].values / 10.0
        
        if self.feature_dim > 2 and 'base_utility' in candidate_ads.columns:
            features[:, 2] = candidate_ads['base_utility'].values
        
        if self.feature_dim > 3:
            is_family = guest_context.get('is_family', False)
            features[:, 3] = float(is_family)
        
        if self.feature_dim > 4:
            guest_price = guest_context.get('price_per_night', 100)
            features[:, 4] = guest_price / 200.0
        
        return features





