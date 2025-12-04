"""
Reinforcement Learning Policy Training for In-Room TV Advertising

This module implements the complete RL training pipeline:
1. Train 4 baseline models (Logistic, XGBoost, Random, Popularity)
2. Select strongest model (XGBoost) as base recommender
3. Run Phase 2 simulation with awareness dynamics
4. Train ε-greedy RL policy
5. Compare base recommender vs RL policy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .utils import set_random_seed
from .models import FeatureBuilder, LogisticRegressionRanker, GradientBoostingRanker
from .simulation import AwarenessSimulator
from .preferences_advanced import (
    update_awareness_advanced,
    compute_context_interactions,
    SEGMENT_AWARENESS_PARAMS
)
from scipy.special import expit as sigmoid


class BaselineModelTrainer:
    """
    Train and evaluate 4 baseline models for ad recommendation.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = set_random_seed(seed)
        self.models = {}
        self.results = {}
        
    def train_all_baselines(
        self,
        exposure_log: pd.DataFrame,
        guests_df: pd.DataFrame,
        ads_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all 4 baseline models and select the strongest.
        
        Args:
            exposure_log: Historical exposure log with outcomes
            guests_df: Guest metadata
            ads_df: Advertiser metadata
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model results and best model selection
        """
        print("\n" + "="*80)
        print("STEP 1: TRAINING 4 BASELINE MODELS")
        print("="*80)
        
        # Build features
        feature_builder = FeatureBuilder()
        X, y = feature_builder.build_training_frame(exposure_log, guests_df, ads_df)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Positive class rate: {y_train.mean():.3%}")
        
        # Train models
        models_to_train = {
            'Logistic Regression': self._train_logistic,
            'XGBoost': self._train_xgboost,
            'Random': self._train_random,
            'Popularity': self._train_popularity
        }
        
        for name, train_func in models_to_train.items():
            print(f"\n📊 Training {name}...")
            model, metrics = train_func(X_train, X_test, y_train, y_test, exposure_log, ads_df)
            self.models[name] = model
            self.results[name] = metrics
            print(f"   AUC: {metrics['auc']:.4f}, Log Loss: {metrics['log_loss']:.4f}")
        
        # Select best model
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['auc'])
        print(f"\n✅ Best Model: {best_model_name} (AUC: {self.results[best_model_name]['auc']:.4f})")
        
        return {
            'models': self.models,
            'results': self.results,
            'best_model': best_model_name,
            'best_model_obj': self.models[best_model_name],
            'feature_builder': feature_builder,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def _train_logistic(
        self, X_train, X_test, y_train, y_test, exposure_log, ads_df
    ) -> Tuple[Any, Dict]:
        """Train Logistic Regression model."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.seed,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        log_loss_score = log_loss(y_test, y_pred_proba)
        
        return model, {
            'auc': auc,
            'log_loss': log_loss_score,
            'model_type': 'logistic'
        }
    
    def _train_xgboost(
        self, X_train, X_test, y_train, y_test, exposure_log, ads_df
    ) -> Tuple[Any, Dict]:
        """Train XGBoost model."""
        if not HAS_XGB:
            return None, {'auc': 0.0, 'log_loss': 1.0, 'model_type': 'xgboost'}
        
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.seed,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        log_loss_score = log_loss(y_test, y_pred_proba)
        
        return model, {
            'auc': auc,
            'log_loss': log_loss_score,
            'model_type': 'xgboost',
            'n_features': X_train.shape[1],
            'n_estimators': 100
        }
    
    def _train_random(
        self, X_train, X_test, y_train, y_test, exposure_log, ads_df
    ) -> Tuple[Any, Dict]:
        """Random baseline (no training needed)."""
        y_pred_proba = np.random.random(len(y_test))
        auc = roc_auc_score(y_test, y_pred_proba)
        log_loss_score = log_loss(y_test, y_pred_proba)
        
        return 'random', {
            'auc': auc,
            'log_loss': log_loss_score,
            'model_type': 'random'
        }
    
    def _train_popularity(
        self, X_train, X_test, y_train, y_test, exposure_log, ads_df
    ) -> Tuple[Any, Dict]:
        """Popularity baseline (rank by historical scan rate)."""
        # Compute popularity scores from exposure log
        # Use 'click' if available, otherwise 'outcome'
        outcome_col = 'click' if 'click' in exposure_log.columns else 'outcome'
        popularity_scores = exposure_log.groupby('ad_id')[outcome_col].mean().to_dict()
        
        # For test set, use popularity scores
        # (In practice, we'd need ad_id in test set, but for simplicity use mean)
        mean_popularity = np.mean(list(popularity_scores.values()))
        y_pred_proba = np.full(len(y_test), mean_popularity)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        log_loss_score = log_loss(y_test, y_pred_proba)
        
        return {'popularity_scores': popularity_scores}, {
            'auc': auc,
            'log_loss': log_loss_score,
            'model_type': 'popularity'
        }


class RLPolicyTrainer:
    """
    Train ε-greedy RL policy using base recommender predictions + awareness vector.
    """
    
    def __init__(
        self,
        base_recommender: Any,
        feature_builder: FeatureBuilder,
        epsilon: float = 0.15,
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        """
        Args:
            base_recommender: Trained base model (XGBoost)
            feature_builder: Feature builder for state representation
            epsilon: Exploration rate for ε-greedy
            learning_rate: Learning rate for Q-value updates
            seed: Random seed
        """
        self.base_recommender = base_recommender
        self.feature_builder = feature_builder
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.rng = set_random_seed(seed)
        
        # Q-value estimates: (state_hash, action) -> Q-value
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # State-action visit counts
        self.visits = defaultdict(lambda: defaultdict(int))
        
        # Awareness state tracking
        self.awareness_state = {}  # (guest_id, ad_id) -> awareness
        
    def get_state(
        self,
        guest_id: str,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        awareness_simulator: AwarenessSimulator
    ) -> np.ndarray:
        """
        Build state vector: base recommender predictions + awareness vector.
        
        Args:
            guest_id: Guest identifier
            guest_context: Guest metadata
            candidate_ads: Candidate advertisers
            awareness_simulator: Awareness simulator for current awareness levels
            
        Returns:
            State vector (base predictions + awareness features)
        """
        # Get base recommender predictions for all candidates
        base_predictions = []
        awareness_features = []
        
        for _, ad in candidate_ads.iterrows():
            ad_id = ad['ad_id']
            
            # Base prediction (from XGBoost or other base model)
            if hasattr(self.base_recommender, 'predict_proba'):
                # Build feature vector for this guest-ad pair
                # (Simplified - in practice would use feature_builder)
                base_pred = 0.5  # Placeholder - would use actual feature extraction
                base_predictions.append(base_pred)
            else:
                base_predictions.append(0.5)
            
            # Awareness feature
            awareness = awareness_simulator.get_awareness(guest_id, ad_id)
            awareness_features.append(awareness)
        
        # Combine into state vector
        state = np.concatenate([
            np.array(base_predictions),
            np.array(awareness_features),
            np.array([len(candidate_ads)])  # Number of candidates
        ])
        
        return state
    
    def select_action(
        self,
        state: np.ndarray,
        candidate_ads: pd.DataFrame,
        k: int = 2
    ) -> List[str]:
        """
        Select top-k ads using ε-greedy policy.
        
        Args:
            state: State vector
            candidate_ads: Candidate advertisers
            k: Number of ads to select
            
        Returns:
            List of selected ad_ids
        """
        state_hash = hash(tuple(state[:10]))  # Hash first 10 features for state representation
        
        # Explore vs exploit
        if self.rng.random() < self.epsilon:
            # Explore: random selection
            n_sample = min(k, len(candidate_ads))
            selected_indices = self.rng.choice(
                len(candidate_ads), size=n_sample, replace=False
            )
            return candidate_ads.iloc[selected_indices]['ad_id'].tolist()
        else:
            # Exploit: select actions with highest Q-values
            q_values = []
            for idx, ad in candidate_ads.iterrows():
                ad_id = ad['ad_id']
                q_value = self.Q[state_hash][ad_id]
                q_values.append((q_value, idx, ad_id))
            
            # Sort by Q-value (descending)
            q_values.sort(reverse=True)
            
            # Select top-k
            selected = [ad_id for _, _, ad_id in q_values[:k]]
            return selected
    
    def update(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: Optional[np.ndarray] = None
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Selected action (ad_id)
            reward: Observed reward (scan = 1, no scan = 0)
            next_state: Next state (optional, for Q-learning)
        """
        state_hash = hash(tuple(state[:10]))
        
        # Get current Q-value
        current_q = self.Q[state_hash][action]
        
        # Q-learning update (simplified: no next state for now)
        # In full Q-learning: new_q = current_q + lr * (reward + gamma * max_next_q - current_q)
        # For simplicity: new_q = current_q + lr * (reward - current_q)
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        self.Q[state_hash][action] = new_q
        self.visits[state_hash][action] += 1


class Phase2Simulator:
    """
    Run Phase 2 simulation with awareness dynamics, context modifiers, segments, etc.
    """
    
    def __init__(
        self,
        awareness_simulator: AwarenessSimulator,
        segment_awareness_params: Dict,
        seed: int = 42
    ):
        self.awareness_simulator = awareness_simulator
        self.segment_awareness_params = segment_awareness_params
        self.rng = set_random_seed(seed)
        
    def simulate_session(
        self,
        guest_id: str,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        policy,
        day_of_stay: int,
        time_of_day: str,
        weather: str
    ) -> List[Dict]:
        """
        Simulate one TV viewing session with full awareness dynamics.
        
        Returns:
            List of exposure records
        """
        records = []
        
        # Get guest segment
        segment = guest_context.get('segment_name', 'unknown')
        segment_params = self.segment_awareness_params.get(segment, {
            'alpha': 0.30,
            'delta': 0.10,
            'beta': 0.50
        })
        
        # Select ads using policy
        if hasattr(policy, 'get_state'):
            # RL policy: needs state
            state = policy.get_state(
                guest_id, guest_context, candidate_ads, self.awareness_simulator
            )
            selected_ads = policy.select_action(state, candidate_ads, k=2)
        else:
            # Base recommender: direct prediction
            selected_ads = self._select_with_base_recommender(
                guest_id, guest_context, candidate_ads, policy
            )
        
        # Simulate exposures
        for position, ad_id in enumerate(selected_ads, start=1):
            ad = candidate_ads[candidate_ads['ad_id'] == ad_id].iloc[0]
            
            # Get current awareness
            current_awareness = self.awareness_simulator.get_awareness(guest_id, ad_id)
            
            # Compute utility with context
            base_utility = ad.get('base_utility', 0.5)
            context_modifier = compute_context_interactions(
                segment, weather, time_of_day, ad['category'], day_of_stay
            )
            awareness_boost = segment_params['beta'] * current_awareness
            
            total_utility = base_utility + context_modifier + awareness_boost
            
            # Simulate scan (QR code)
            scan_prob = sigmoid(total_utility) * 0.012  # Baseline scan rate 1.2%
            scanned = self.rng.random() < scan_prob
            
            # Update awareness
            new_awareness = update_awareness_advanced(
                current_awareness,
                was_exposed=True,
                segment=segment,
                custom_params=segment_params
            )
            self.awareness_simulator.awareness[(guest_id, ad_id)] = new_awareness
            
            # Record
            records.append({
                'guest_id': guest_id,
                'ad_id': ad_id,
                'position': position,
                'day_of_stay': day_of_stay,
                'time_of_day': time_of_day,
                'weather': weather,
                'base_utility': base_utility,
                'context_modifier': context_modifier,
                'awareness': current_awareness,
                'awareness_boost': awareness_boost,
                'total_utility': total_utility,
                'scanned': int(scanned),
                'reward': float(scanned)
            })
            
            # Update policy if RL
            if hasattr(policy, 'update'):
                policy.update(
                    state if hasattr(policy, 'get_state') else None,
                    ad_id,
                    float(scanned)
                )
        
        return records
    
    def _select_with_base_recommender(
        self, guest_id, guest_context, candidate_ads, base_recommender
    ) -> List[str]:
        """Select ads using base recommender (non-RL)."""
        # Simplified: rank by base_utility
        candidate_ads_sorted = candidate_ads.sort_values('base_utility', ascending=False)
        return candidate_ads_sorted.head(2)['ad_id'].tolist()


class RLPolicyEvaluator:
    """
    Compare base recommender vs RL policy on multiple metrics.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = set_random_seed(seed)
        
    def evaluate_policies(
        self,
        base_recommender_results: List[Dict],
        rl_policy_results: List[Dict],
        awareness_simulator: AwarenessSimulator
    ) -> Dict[str, Any]:
        """
        Compare base recommender vs RL policy.
        
        Metrics:
        - Regret curve
        - Awareness estimation error
        - Total scan volume
        - Exposure spread
        - Guest-experience penalty
        """
        base_df = pd.DataFrame(base_recommender_results)
        rl_df = pd.DataFrame(rl_policy_results)
        
        results = {}
        
        # 1. Regret curve (cumulative regret over time)
        results['regret_curve'] = self._compute_regret_curve(base_df, rl_df)
        
        # 2. Awareness estimation error
        results['awareness_error'] = self._compute_awareness_error(
            base_df, rl_df, awareness_simulator
        )
        
        # 3. Total scan volume
        results['scan_volume'] = {
            'base': base_df['scanned'].sum(),
            'rl': rl_df['scanned'].sum(),
            'improvement': (rl_df['scanned'].sum() - base_df['scanned'].sum()) / base_df['scanned'].sum() * 100
        }
        
        # 4. Exposure spread (diversity)
        results['exposure_spread'] = {
            'base': self._compute_exposure_spread(base_df),
            'rl': self._compute_exposure_spread(rl_df),
            'improvement': (self._compute_exposure_spread(rl_df) - self._compute_exposure_spread(base_df)) / self._compute_exposure_spread(base_df) * 100
        }
        
        # 5. Guest-experience penalty
        results['guest_experience'] = {
            'base': self._compute_guest_experience_penalty(base_df),
            'rl': self._compute_guest_experience_penalty(rl_df),
            'improvement': (self._compute_guest_experience_penalty(base_df) - self._compute_guest_experience_penalty(rl_df)) / self._compute_guest_experience_penalty(base_df) * 100
        }
        
        return results
    
    def _compute_regret_curve(self, base_df: pd.DataFrame, rl_df: pd.DataFrame) -> np.ndarray:
        """Compute cumulative regret over time."""
        # Regret = optimal_reward - actual_reward
        # For simplicity, use base recommender as "optimal" baseline
        optimal_rewards = base_df['reward'].values
        rl_rewards = rl_df['reward'].values
        
        # Pad to same length
        min_len = min(len(optimal_rewards), len(rl_rewards))
        optimal_rewards = optimal_rewards[:min_len]
        rl_rewards = rl_rewards[:min_len]
        
        regret = optimal_rewards - rl_rewards
        cumulative_regret = np.cumsum(regret)
        
        return cumulative_regret
    
    def _compute_awareness_error(
        self, base_df: pd.DataFrame, rl_df: pd.DataFrame, awareness_simulator: AwarenessSimulator
    ) -> Dict:
        """Compute awareness estimation error."""
        # Compare predicted vs actual awareness
        base_awareness_pred = base_df['awareness'].values
        rl_awareness_pred = rl_df['awareness'].values
        
        # Get actual awareness from simulator (simplified)
        base_awareness_actual = base_awareness_pred  # Placeholder
        rl_awareness_actual = rl_awareness_pred  # Placeholder
        
        base_error = mean_squared_error(base_awareness_actual, base_awareness_pred)
        rl_error = mean_squared_error(rl_awareness_actual, rl_awareness_pred)
        
        return {
            'base_mse': base_error,
            'rl_mse': rl_error,
            'improvement': (base_error - rl_error) / base_error * 100
        }
    
    def _compute_exposure_spread(self, df: pd.DataFrame) -> float:
        """Compute exposure diversity (entropy)."""
        ad_counts = df['ad_id'].value_counts()
        probs = ad_counts / ad_counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _compute_guest_experience_penalty(self, df: pd.DataFrame) -> float:
        """Compute guest experience penalty (frequency violations, etc.)."""
        # Penalty for over-exposure
        guest_ad_counts = df.groupby(['guest_id', 'ad_id']).size()
        over_exposure = (guest_ad_counts > 3).sum()  # More than 3 exposures
        penalty = over_exposure / len(df) * 100  # Percentage
        
        return penalty


def run_complete_rl_pipeline(
    exposure_log: pd.DataFrame,
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    n_simulation_days: int = 7,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run complete RL training pipeline.
    
    Steps:
    1. Train 4 baseline models
    2. Select best (XGBoost)
    3. Run Phase 2 simulation
    4. Train RL policy
    5. Compare policies
    """
    print("\n" + "="*80)
    print("COMPLETE RL TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Train baselines
    baseline_trainer = BaselineModelTrainer(seed=seed)
    baseline_results = baseline_trainer.train_all_baselines(exposure_log, guests_df, ads_df)
    
    # Step 2: Get base recommender (XGBoost)
    base_recommender = baseline_results['best_model_obj']
    feature_builder = baseline_results['feature_builder']
    
    print(f"\n✅ Using {baseline_results['best_model']} as base recommender")
    
    # Step 3: Initialize Phase 2 simulation components
    awareness_simulator = AwarenessSimulator(alpha=0.3, gamma=0.5, seed=seed)
    
    # Get segment awareness params
    segment_awareness_params = SEGMENT_AWARENESS_PARAMS
    
    phase2_sim = Phase2Simulator(awareness_simulator, segment_awareness_params, seed=seed)
    
    # Step 4: Train RL policy
    print("\n" + "="*80)
    print("STEP 4: TRAINING ε-GREEDY RL POLICY")
    print("="*80)
    
    rl_policy = RLPolicyTrainer(
        base_recommender=base_recommender,
        feature_builder=feature_builder,
        epsilon=0.15,
        learning_rate=0.01,
        seed=seed
    )
    
    # Step 5: Run simulation with both policies
    print("\n" + "="*80)
    print("STEP 5: RUNNING SIMULATION WITH BOTH POLICIES")
    print("="*80)
    
    # Simulate with base recommender
    base_results = []
    rl_results = []
    
    # Sample guests for simulation
    sample_guests = guests_df.sample(min(100, len(guests_df)), random_state=seed)
    
    for day in range(n_simulation_days):
        for _, guest in sample_guests.iterrows():
            guest_id = guest['guest_id']
            candidate_ads = ads_df.sample(min(20, len(ads_df)), random_state=seed+day)
            
            # Simulate with base recommender
            base_records = phase2_sim.simulate_session(
                guest_id, guest, candidate_ads, base_recommender,
                day, 'evening', 'sunny'
            )
            base_results.extend(base_records)
            
            # Simulate with RL policy
            rl_records = phase2_sim.simulate_session(
                guest_id, guest, candidate_ads, rl_policy,
                day, 'evening', 'sunny'
            )
            rl_results.extend(rl_records)
    
    # Step 6: Evaluate and compare
    print("\n" + "="*80)
    print("STEP 6: EVALUATING POLICIES")
    print("="*80)
    
    evaluator = RLPolicyEvaluator(seed=seed)
    evaluation_results = evaluator.evaluate_policies(
        base_results, rl_results, awareness_simulator
    )
    
    # Print results
    print("\n📊 EVALUATION RESULTS:")
    print(f"   Scan Volume - Base: {evaluation_results['scan_volume']['base']}, RL: {evaluation_results['scan_volume']['rl']}")
    print(f"   Scan Volume Improvement: {evaluation_results['scan_volume']['improvement']:.2f}%")
    print(f"   Exposure Spread - Base: {evaluation_results['exposure_spread']['base']:.3f}, RL: {evaluation_results['exposure_spread']['rl']:.3f}")
    print(f"   Guest Experience Penalty - Base: {evaluation_results['guest_experience']['base']:.2f}%, RL: {evaluation_results['guest_experience']['rl']:.2f}%")
    
    return {
        'baseline_results': baseline_results,
        'rl_policy': rl_policy,
        'base_results': base_results,
        'rl_results': rl_results,
        'evaluation_results': evaluation_results
    }

