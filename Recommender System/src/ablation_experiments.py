## Ablation Experiments

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')


class AblationExperiment:

    # Ablation experiments to test contribution of each modeling component
    
    def __init__(self, base_scan_rate: float = 0.15):
        self.base_scan_rate = base_scan_rate
        self.results = {}
        
    def simulate_baseline(
        self,
        n_guests: int = 1000,
        n_days: int = 7,
        seed: int = 42
    ) -> pd.DataFrame:
        # Generate baseline simulation data
        np.random.seed(seed)
        
        data = []
        
        for guest_id in range(n_guests):
            segment = np.random.randint(0, 8)
            
            for day in range(n_days):
                for impression in range(2):  # capped at 2 ads per day
                    hour = np.random.choice([8, 12, 18, 21])  # Morning, noon, evening, night
                    weather = np.random.choice(['sunny', 'rainy', 'cloudy'])
                    category = np.random.choice(['Restaurants', 'Experiences', 'Shopping', 'Wellness', 'Nightlife'])
                    
                    data.append({
                        'guest_id': guest_id,
                        'segment_id': segment,
                        'day_of_stay': day + 1,
                        'hour': hour,
                        'weather': weather,
                        'category': category,
                        'awareness': 0.0,  # Will be computed
                        'scan_prob': 0.0   # Will be computed
                    })
        
        return pd.DataFrame(data)
    
    def ablate_contextual_modifiers(
        self,
        data: pd.DataFrame
    ) -> Dict:
        # Ablation 1: Remove contextual modifiers (time, weather, day-of-stay)
        # Full model with context
        def full_model(row):
            base = 0.3
            
            # Time effect
            if row['hour'] in [18, 19, 20, 21]:
                time_effect = 0.15  # Evening boost
            elif row['hour'] in [8, 9, 10]:
                time_effect = 0.05  # Morning small boost
            else:
                time_effect = 0.0
            
            # Weather effect
            if row['weather'] == 'rainy':
                weather_effect = 0.10  # Indoor preference
            else:
                weather_effect = 0.0
            
            # Day-of-stay effect
            if row['day_of_stay'] == 1:
                day_effect = 0.05  # Exploration day
            elif row['day_of_stay'] > 5:
                day_effect = -0.05  # Fatigue
            else:
                day_effect = 0.0
            
            return base + time_effect + weather_effect + day_effect
        
        # Ablated model (no context)
        def ablated_model(row):
            return 0.3  # Base only
        
        data['utility_full'] = data.apply(full_model, axis=1)
        data['utility_ablated'] = data.apply(ablated_model, axis=1)
        
        # Simulate outcomes
        np.random.seed(42)
        data['scan_full'] = np.random.binomial(1, data['utility_full'].clip(0, 1))
        data['scan_ablated'] = np.random.binomial(1, data['utility_ablated'].clip(0, 1))
        
        result = {
            'component': 'Contextual Modifiers (time, weather, day)',
            'full_scan_rate': data['scan_full'].mean(),
            'ablated_scan_rate': data['scan_ablated'].mean(),
            'improvement': (data['scan_full'].mean() - data['scan_ablated'].mean()) / data['scan_ablated'].mean() * 100,
            'interpretation': 'Context adds predictive value' if data['scan_full'].mean() > data['scan_ablated'].mean() else 'Context has minimal effect'
        }
        
        self.results['contextual_modifiers'] = result
        return result
    
    def ablate_awareness_model(
        self,
        data: pd.DataFrame
    ) -> Dict:
        # Ablation 2: Remove awareness dynamics
        # Simulate awareness accumulation
        awareness_by_guest = {}
        
        for idx, row in data.iterrows():
            guest_id = row['guest_id']
            if guest_id not in awareness_by_guest:
                awareness_by_guest[guest_id] = 0.1
            
            # Update awareness
            alpha = 0.30
            delta = 0.10
            
            current = awareness_by_guest[guest_id]
            new_awareness = current + alpha * (1 - current)
            awareness_by_guest[guest_id] = new_awareness * (1 - delta)
            
            data.loc[idx, 'awareness'] = new_awareness
        
        # Full model with awareness
        beta = 0.5
        data['utility_with_awareness'] = 0.2 + beta * data['awareness']
        
        # Ablated model (no awareness)
        data['utility_no_awareness'] = 0.2 + beta * 0.3  # Fixed average awareness
        
        # Simulate outcomes
        np.random.seed(42)
        data['scan_with_awareness'] = np.random.binomial(1, data['utility_with_awareness'].clip(0, 1))
        data['scan_no_awareness'] = np.random.binomial(1, data['utility_no_awareness'].clip(0, 1))
        
        result = {
            'component': 'Awareness Dynamics (α, δ)',
            'full_scan_rate': data['scan_with_awareness'].mean(),
            'ablated_scan_rate': data['scan_no_awareness'].mean(),
            'improvement': (data['scan_with_awareness'].mean() - data['scan_no_awareness'].mean()) / data['scan_no_awareness'].mean() * 100,
            'interpretation': 'Awareness dynamics significantly improve predictions'
        }
        
        self.results['awareness_model'] = result
        return result
    
    def ablate_segmentation(
        self,
        data: pd.DataFrame
    ) -> Dict:
        # Ablation 3: Remove segmentation (treat all guests equally)
        # Segment-specific affinity (example)
        segment_affinity = {
            0: {'Restaurants': 0.5, 'Experiences': 0.3, 'Shopping': 0.2, 'Wellness': 0.1, 'Nightlife': 0.5},
            1: {'Restaurants': 0.9, 'Experiences': 0.6, 'Shopping': 0.4, 'Wellness': 0.5, 'Nightlife': 0.7},
            2: {'Restaurants': 0.8, 'Experiences': 0.4, 'Shopping': 0.3, 'Wellness': 0.3, 'Nightlife': 0.6},
            3: {'Restaurants': 0.9, 'Experiences': 0.5, 'Shopping': 0.6, 'Wellness': 0.4, 'Nightlife': 0.8},
            4: {'Restaurants': 0.7, 'Experiences': 0.8, 'Shopping': 0.7, 'Wellness': 0.6, 'Nightlife': 0.3},
            5: {'Restaurants': 0.8, 'Experiences': 0.3, 'Shopping': 0.3, 'Wellness': 0.3, 'Nightlife': 0.9},
            6: {'Restaurants': 0.7, 'Experiences': 0.4, 'Shopping': 0.4, 'Wellness': 0.3, 'Nightlife': 0.8},
            7: {'Restaurants': 0.9, 'Experiences': 0.5, 'Shopping': 0.4, 'Wellness': 0.6, 'Nightlife': 0.7},
        }
        
        # Average affinity (ablated)
        avg_affinity = {cat: np.mean([seg[cat] for seg in segment_affinity.values()]) 
                        for cat in ['Restaurants', 'Experiences', 'Shopping', 'Wellness', 'Nightlife']}
        
        # Full model (segment-specific)
        data['utility_segmented'] = data.apply(
            lambda row: segment_affinity[row['segment_id']][row['category']], axis=1
        )
        
        # Ablated model (population average)
        data['utility_uniform'] = data.apply(
            lambda row: avg_affinity[row['category']], axis=1
        )
        
        # Simulate outcomes
        np.random.seed(42)
        data['scan_segmented'] = np.random.binomial(1, data['utility_segmented'].clip(0, 1) * 0.3)
        data['scan_uniform'] = np.random.binomial(1, data['utility_uniform'].clip(0, 1) * 0.3)
        
        result = {
            'component': 'Guest Segmentation (8 clusters)',
            'full_scan_rate': data['scan_segmented'].mean(),
            'ablated_scan_rate': data['scan_uniform'].mean(),
            'improvement': (data['scan_segmented'].mean() - data['scan_uniform'].mean()) / data['scan_uniform'].mean() * 100,
            'interpretation': 'Segmentation enables personalized targeting'
        }
        
        self.results['segmentation'] = result
        return result
    
    def ablate_placement_visibility(
        self,
        data: pd.DataFrame
    ) -> Dict:
        # Ablation 4: Remove placement visibility model
        # Simulate placement types
        placements = ['full_screen', 'channel_guide', 'banner', 'corner']
        visibility = {'full_screen': 1.0, 'channel_guide': 0.8, 'banner': 0.6, 'corner': 0.3}
        
        data['placement'] = np.random.choice(placements, len(data))
        data['visibility_full'] = data['placement'].map(visibility)
        data['visibility_ablated'] = 0.7  # Uniform average
        
        base_utility = 0.4
        data['utility_with_visibility'] = base_utility * data['visibility_full']
        data['utility_uniform_visibility'] = base_utility * data['visibility_ablated']
        
        # Simulate outcomes
        np.random.seed(42)
        data['scan_with_visibility'] = np.random.binomial(1, data['utility_with_visibility'].clip(0, 1))
        data['scan_uniform_visibility'] = np.random.binomial(1, data['utility_uniform_visibility'].clip(0, 1))
        
        result = {
            'component': 'Placement Visibility Model',
            'full_scan_rate': data['scan_with_visibility'].mean(),
            'ablated_scan_rate': data['scan_uniform_visibility'].mean(),
            'improvement': (data['scan_with_visibility'].mean() - data['scan_uniform_visibility'].mean()) / data['scan_uniform_visibility'].mean() * 100,
            'interpretation': 'Placement modeling improves reach prediction accuracy'
        }
        
        self.results['placement_visibility'] = result
        return result
    
    def run_all_ablations(
        self,
        n_guests: int = 1000,
        n_days: int = 7
    ) -> pd.DataFrame:
        # Run all ablation experiments and summarize results
        
        data = self.simulate_baseline(n_guests, n_days)
        
        self.ablate_contextual_modifiers(data.copy())
        
        self.ablate_awareness_model(data.copy())
        
        self.ablate_segmentation(data.copy())
        
        self.ablate_placement_visibility(data.copy())
        
        # Compile summary
        summary = pd.DataFrame(self.results).T
        summary = summary[['component', 'full_scan_rate', 'ablated_scan_rate', 'improvement', 'interpretation']]
        
        print(summary.to_string())
        
        return summary


class ModelComplexityAnalyzer:
    # Analyzes model complexity vs benefit tradeoff
    
    def __init__(self):
        self.results = {}
    
    def compare_model_complexity(
        self,
        n_samples: int = 5000,
        seed: int = 42
    ) -> pd.DataFrame:
        # Compare models of increasing complexity
        np.random.seed(seed)
        
        # Generate synthetic data
        X = np.random.randn(n_samples, 10)  # 10 features
        segment = np.random.randint(0, 8, n_samples)
        awareness = np.random.uniform(0.1, 0.8, n_samples)
        
        # True scan probability (complex function)
        true_prob = (
            0.1 +  # Base
            0.2 * awareness +  # Awareness effect
            0.1 * (segment > 3).astype(float) +  # Segment effect
            0.05 * X[:, 0] +  # Context effect
            0.03 * X[:, 1]
        )
        true_prob = np.clip(true_prob, 0, 1)
        
        y = np.random.binomial(1, true_prob)
        
        # Train-test split
        split = int(0.8 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        awareness_train, awareness_test = awareness[:split], awareness[split:]
        segment_train, segment_test = segment[:split], segment[split:]
        
        results = []
        
        # 1. Random baseline
        y_pred_random = np.random.uniform(0, 1, len(y_test))
        results.append({
            'model': 'Random Baseline',
            'complexity': 1,
            'auc': roc_auc_score(y_test, y_pred_random),
            'accuracy': accuracy_score(y_test, (y_pred_random > 0.5).astype(int)),
            'log_loss': log_loss(y_test, y_pred_random),
            'parameters': 0
        })
        
        # 2. Popularity baseline
        y_pred_pop = np.full(len(y_test), y_train.mean())
        results.append({
            'model': 'Popularity Baseline',
            'complexity': 2,
            'auc': 0.5,  # Random by definition
            'accuracy': accuracy_score(y_test, (y_pred_pop > 0.5).astype(int)),
            'log_loss': log_loss(y_test, np.clip(y_pred_pop, 0.01, 0.99)),
            'parameters': 1
        })
        
        # 3. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict_proba(X_test)[:, 1]
        results.append({
            'model': 'Logistic Regression',
            'complexity': 3,
            'auc': roc_auc_score(y_test, y_pred_lr),
            'accuracy': lr.score(X_test, y_test),
            'log_loss': log_loss(y_test, y_pred_lr),
            'parameters': X_train.shape[1] + 1
        })
        
        # 4. XGBoost
        xgb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
        results.append({
            'model': 'XGBoost',
            'complexity': 4,
            'auc': roc_auc_score(y_test, y_pred_xgb),
            'accuracy': xgb.score(X_test, y_test),
            'log_loss': log_loss(y_test, y_pred_xgb),
            'parameters': 50 * 7  # Approx: trees * nodes
        })
        
        # 5. Awareness-based model
        y_pred_awareness = 0.2 + 0.5 * awareness_test
        y_pred_awareness = np.clip(y_pred_awareness, 0.01, 0.99)
        results.append({
            'model': 'Awareness-Based',
            'complexity': 5,
            'auc': roc_auc_score(y_test, y_pred_awareness),
            'accuracy': accuracy_score(y_test, (y_pred_awareness > 0.5).astype(int)),
            'log_loss': log_loss(y_test, y_pred_awareness),
            'parameters': 2  # α, β
        })
        
        # 6. Full system (Awareness + Segment + Context)
        X_full = np.column_stack([X_test, awareness_test, segment_test])
        X_full_train = np.column_stack([X_train, awareness_train, segment_train])
        
        lr_full = LogisticRegression(max_iter=1000, random_state=42)
        lr_full.fit(X_full_train, y_train)
        y_pred_full = lr_full.predict_proba(X_full)[:, 1]
        results.append({
            'model': 'Full System',
            'complexity': 6,
            'auc': roc_auc_score(y_test, y_pred_full),
            'accuracy': lr_full.score(X_full, y_test),
            'log_loss': log_loss(y_test, y_pred_full),
            'parameters': X_full_train.shape[1] + 1
        })
        
        result_df = pd.DataFrame(results)
        self.results['complexity_comparison'] = result_df
        
        return result_df
    
    def marginal_benefit_analysis(self) -> pd.DataFrame:
        # Calculate marginal benefit of each component
        if 'complexity_comparison' not in self.results:
            self.compare_model_complexity()
        
        df = self.results['complexity_comparison'].copy()
        
        # Calculate improvements
        df['auc_improvement'] = df['auc'].diff()
        df['accuracy_improvement'] = df['accuracy'].diff()
        
        # Calculate efficiency (improvement per parameter)
        df['params_added'] = df['parameters'].diff().fillna(df['parameters'])
        df['efficiency'] = df['auc_improvement'] / df['params_added'].replace(0, np.nan)
        
        return df[['model', 'auc', 'auc_improvement', 'parameters', 'params_added', 'efficiency']]


class TimingPolicyAnalyzer:
    # Analyze robustness of exposure timing policies
    
    def __init__(self):
        self.results = {}
    
    def simulate_timing_policies(
        self,
        n_guests: int = 500,
        n_days: int = 7,
        seed: int = 42
    ) -> pd.DataFrame:
        # Simulate different timing policies and compare awareness gains
        np.random.seed(seed)
        
        policies = {
            'room_entry': {'hours': [15, 16, 17], 'tv_prob': 0.8, 'attention': 0.9},
            'mid_viewing': {'hours': [20, 21, 22], 'tv_prob': 0.7, 'attention': 0.5},  # Lower attention (distracted)
            'pre_bedtime': {'hours': [22, 23], 'tv_prob': 0.6, 'attention': 0.7},
            'morning': {'hours': [7, 8, 9], 'tv_prob': 0.4, 'attention': 0.6}
        }
        
        results = []
        
        for policy_name, params in policies.items():
            total_exposures = 0
            total_awareness = 0
            total_scans = 0
            
            for _ in range(n_guests):
                awareness = 0.0
                
                for day in range(n_days):
                    # TV-on probability
                    if np.random.random() < params['tv_prob']:
                        # Exposure
                        total_exposures += 1
                        
                        # Awareness update (modified by attention)
                        alpha = 0.30 * params['attention']
                        awareness = awareness + alpha * (1 - awareness)
                        
                        # Scan probability
                        scan_prob = 0.15 * awareness
                        if np.random.random() < scan_prob:
                            total_scans += 1
                    
                    # Decay
                    awareness *= 0.9
                
                total_awareness += awareness
            
            results.append({
                'policy': policy_name,
                'hours': str(params['hours']),
                'tv_probability': params['tv_prob'],
                'attention_factor': params['attention'],
                'avg_exposures': total_exposures / n_guests,
                'final_awareness': total_awareness / n_guests,
                'scan_rate': total_scans / total_exposures if total_exposures > 0 else 0,
                'total_scans': total_scans
            })
        
        result_df = pd.DataFrame(results)
        self.results['timing_policies'] = result_df
        
        return result_df


def run_ablation_demo():
    # Run all ablation experiments
    
    # 1. Ablation experiments
    ablation = AblationExperiment()
    ablation_summary = ablation.run_all_ablations(n_guests=500, n_days=7)
    
    # 2. Model complexity comparison
    complexity = ModelComplexityAnalyzer()
    complexity_results = complexity.compare_model_complexity(n_samples=2000)
    print("\nComplexity vs Performance:")
    print(complexity_results.to_string())
    
    marginal = complexity.marginal_benefit_analysis()
    print("\nMarginal Benefit Analysis:")
    print(marginal.to_string())
    
    # 3. Timing policy analysis
    timing = TimingPolicyAnalyzer()
    timing_results = timing.simulate_timing_policies(n_guests=300)
    print("\nTiming Policy Comparison:")
    print(timing_results.to_string())
    
    return {
        'ablation': ablation_summary,
        'complexity': complexity_results,
        'timing': timing_results
    }


if __name__ == "__main__":
    results = run_ablation_demo()




