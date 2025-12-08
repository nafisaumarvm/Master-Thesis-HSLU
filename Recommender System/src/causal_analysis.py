# Causal Analysis Module for In-Room TV Advertising


import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Formal definition of popularity baseline
class PopularityBaseline:
    # Formal definition of popularity baseline
    
    def __init__(self, impressions: pd.DataFrame):
        self.impressions = impressions
        self.baselines = {}
        
    def compute_impression_popularity(self) -> pd.Series:
        # Naive popularity: proportion of total impressions
        U_pop(c) = Impressions(c) / Σ_c' Impressions(c')
        category_counts = self.impressions['category'].value_counts()
        self.baselines['impression_popularity'] = category_counts / category_counts.sum()
        return self.baselines['impression_popularity']
    
    def compute_engagement_popularity(self) -> pd.Series:
        # Observed scan rate per category
        U_pop(c) = E[scan_i | c] = (Scans in c) / (Impressions in c)
        category_engagement = self.impressions.groupby('category').agg({
            'scanned': ['sum', 'count']
        })
        category_engagement.columns = ['scans', 'impressions']
        category_engagement['engagement_rate'] = category_engagement['scans'] / category_engagement['impressions']
        self.baselines['engagement_popularity'] = category_engagement['engagement_rate']
        return self.baselines['engagement_popularity']
    
    def compute_corrected_popularity(self, propensity_weights: pd.Series) -> pd.Series:
        # Inverse propensity weighted popularity
        U_pop_IPW(c) = Σ_i (w_i * scan_i * 1[c_i = c]) / Σ_i (w_i * 1[c_i = c])
        df = self.impressions.copy()
        df['weight'] = propensity_weights
        df['weighted_scan'] = df['scanned'] * df['weight']
        
        category_weighted = df.groupby('category').agg({
            'weighted_scan': 'sum',
            'weight': 'sum'
        })
        category_weighted['ipw_rate'] = category_weighted['weighted_scan'] / category_weighted['weight']
        self.baselines['corrected_popularity'] = category_weighted['ipw_rate']
        return self.baselines['corrected_popularity']
    
    def get_baseline_comparison(self) -> pd.DataFrame:
        # Return comparison of all computed baselines
        return pd.DataFrame(self.baselines)


# Endogeneity analysis
class EndogeneityAnalyzer:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.endogeneity_results = {}
    
    def test_exposure_randomness(self) -> Dict:
        # Test whether exposure is quasi-random
        results = {}
        
        # Test 1: Balance check on observed covariates
        if 'exposed' in self.data.columns and 'segment_id' in self.data.columns:
            contingency = pd.crosstab(self.data['exposed'], self.data['segment_id'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results['segment_balance'] = {
                'chi2': chi2,
                'p_value': p_value,
                'interpretation': 'Random' if p_value > 0.05 else 'Non-random'
            }
        
        # Test 2: Time-of-day correlation with exposure
        if 'hour' in self.data.columns and 'exposed' in self.data.columns:
            correlation = self.data['hour'].corr(self.data['exposed'].astype(float))
            results['time_correlation'] = {
                'correlation': correlation,
                'interpretation': 'Weak' if abs(correlation) < 0.3 else 'Strong'
            }
        
        self.endogeneity_results['randomness_test'] = results
        return results
    
    def identify_instruments(self) -> List[str]:
        # Identify valid instrumental variables for exposure
        instruments = []
        
        # Time-of-entry as IV
        if 'entry_hour' in self.data.columns:
            # Check relevance: does entry_hour predict exposure?
            if 'exposed' in self.data.columns:
                corr = self.data['entry_hour'].corr(self.data['exposed'].astype(float))
                if abs(corr) > 0.1:
                    instruments.append('entry_hour')
        
        # Weather as IV
        if 'weather_rain' in self.data.columns:
            instruments.append('weather_rain')
        
        # Day of week as IV
        if 'day_of_week' in self.data.columns:
            instruments.append('day_of_week')
        
        self.endogeneity_results['instruments'] = instruments
        return instruments
    
    def compare_endogeneity_sources(self) -> pd.DataFrame:
        # Compare endogeneity in web vs. in-room TV advertising
        comparison = pd.DataFrame({
            'Source': [
                'Algorithmic selection',
                'User self-selection',
                'Position bias',
                'Popularity bias',
                'Timing effects'
            ],
            'Web_Advertising': [
                'Severe (personalized ranking)',
                'Severe (click to see)',
                'Severe (ordered lists)',
                'Severe (viral effects)',
                'Moderate (browsing patterns)'
            ],
            'InRoom_TV': [
                'Weak (entry-time based)',
                'Weak (captive audience)',
                'Moderate (startup placement)',
                'Weak (no viral spread)',
                'Moderate (viewing habits)'
            ],
            'Mitigation': [
                'Random assignment at entry',
                'N/A (inherent exposure)',
                'Full-screen startup',
                'Frequency caps',
                'IV: entry time, weather'
            ]
        })
        return comparison

# Causal effect estimator

class CausalEffectEstimator:
    # Estimates causal effects of exposure on awareness and engagement

    def __init__(self, data: pd.DataFrame):
        # Estimates causal effects of exposure on awareness and engagement
        self.data = data
        self.results = {}
    
    def estimate_propensity_scores(
        self, 
        treatment_col: str = 'exposed',
        covariates: List[str] = None
    ) -> pd.Series:
        # Estimate propensity scores P(exposed | X)
        if covariates is None:
            covariates = ['segment_id', 'day_of_stay', 'hour']
        
        available_covariates = [c for c in covariates if c in self.data.columns]
        
        if not available_covariates:
            # Return uniform propensity if no covariates
            return pd.Series(0.5, index=self.data.index)
        
        X = pd.get_dummies(self.data[available_covariates], drop_first=True)
        y = self.data[treatment_col].astype(int)
        
        # Fit logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        propensity_scores = model.predict_proba(X)[:, 1]
        
        # Clip to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
        
        self.results['propensity_scores'] = propensity_scores
        return pd.Series(propensity_scores, index=self.data.index)
    
    def estimate_ATE_naive(
        self, 
        treatment_col: str = 'exposed',
        outcome_col: str = 'scanned'
    ) -> Dict:
        # Naive ATE: E[Y|T=1] - E[Y|T=0]
        
        treated = self.data[self.data[treatment_col] == 1][outcome_col]
        control = self.data[self.data[treatment_col] == 0][outcome_col]
        
        ate_naive = treated.mean() - control.mean()
        
        # Standard error
        se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
        
        result = {
            'ATE': ate_naive,
            'SE': se,
            'CI_lower': ate_naive - 1.96 * se,
            'CI_upper': ate_naive + 1.96 * se,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'n_treated': len(treated),
            'n_control': len(control),
            'method': 'Naive (biased)'
        }
        
        self.results['ATE_naive'] = result
        return result
    
    def estimate_ATE_IPW(
        self,
        treatment_col: str = 'exposed',
        outcome_col: str = 'scanned',
        propensity_scores: pd.Series = None
    ) -> Dict:
        # IPW-corrected ATE

        if propensity_scores is None:
            propensity_scores = self.estimate_propensity_scores(treatment_col)
        
        T = self.data[treatment_col].values
        Y = self.data[outcome_col].values
        e = propensity_scores.values
        
        # IPW estimator
        treated_term = (Y * T / e).mean()
        control_term = (Y * (1 - T) / (1 - e)).mean()
        ate_ipw = treated_term - control_term
        
        # Bootstrap SE
        n_bootstrap = 200
        bootstrap_ates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), len(Y), replace=True)
            T_b, Y_b, e_b = T[idx], Y[idx], e[idx]
            ate_b = (Y_b * T_b / e_b).mean() - (Y_b * (1 - T_b) / (1 - e_b)).mean()
            bootstrap_ates.append(ate_b)
        
        se = np.std(bootstrap_ates)
        
        result = {
            'ATE': ate_ipw,
            'SE': se,
            'CI_lower': ate_ipw - 1.96 * se,
            'CI_upper': ate_ipw + 1.96 * se,
            'method': 'IPW (corrected)'
        }
        
        self.results['ATE_IPW'] = result
        return result
    
    def estimate_dose_response(
        self,
        exposure_col: str = 'exposure_count',
        outcome_col: str = 'scanned',
        max_exposure: int = 10
    ) -> pd.DataFrame:
        # Estimate dose-response curve: E[Y | Exposure = k]
        dose_response = []
        
        for k in range(max_exposure + 1):
            if exposure_col in self.data.columns:
                subset = self.data[self.data[exposure_col] == k]
            else:
                # Simulate exposure counts
                subset = self.data.sample(frac=0.1, random_state=k)
            
            if len(subset) > 10:
                mean_outcome = subset[outcome_col].mean()
                se = subset[outcome_col].std() / np.sqrt(len(subset))
                
                dose_response.append({
                    'exposure': k,
                    'mean_outcome': mean_outcome,
                    'se': se,
                    'ci_lower': mean_outcome - 1.96 * se,
                    'ci_upper': mean_outcome + 1.96 * se,
                    'n': len(subset)
                })
        
        result = pd.DataFrame(dose_response)
        self.results['dose_response'] = result
        return result
    
    def estimate_awareness_causal_effect(
        self,
        awareness_col: str = 'awareness',
        outcome_col: str = 'scanned'
    ) -> Dict:
        # Estimate causal effect of awareness on engagement.

        if awareness_col not in self.data.columns:
            return {'error': 'Awareness column not found'}
        
        # Logistic regression
        X = self.data[[awareness_col]].values
        if 'segment_id' in self.data.columns:
            X = np.column_stack([X, pd.get_dummies(self.data['segment_id']).values])
        
        y = self.data[outcome_col].values
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Awareness coefficient
        awareness_coef = model.coef_[0][0]
        
        # Marginal effect at mean
        mean_awareness = self.data[awareness_col].mean()
        prob_at_mean = model.predict_proba(X.mean(axis=0).reshape(1, -1))[0, 1]
        marginal_effect = awareness_coef * prob_at_mean * (1 - prob_at_mean)
        
        result = {
            'awareness_coefficient': awareness_coef,
            'marginal_effect': marginal_effect,
            'interpretation': f'1 unit awareness increase → {marginal_effect:.3f} increase in scan probability',
            'model_accuracy': model.score(X, y)
        }
        
        self.results['awareness_effect'] = result
        return result


# Robustness and sensitivity analysis

class RobustnessAnalyzer:
    # Robustness checks for awareness parameters

    def __init__(self, base_alpha: float = 0.30, base_delta: float = 0.10):
        self.base_alpha = base_alpha
        self.base_delta = base_delta
        self.results = {}
    
    def sensitivity_analysis(
        self,
        alpha_range: Tuple[float, float] = (0.1, 0.5),
        delta_range: Tuple[float, float] = (0.02, 0.20),
        n_steps: int = 10,
        n_simulations: int = 100
    ) -> pd.DataFrame:
        # Sensitivity analysis across parameter grid

        alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
        deltas = np.linspace(delta_range[0], delta_range[1], n_steps)
        
        results = []
        
        for alpha in alphas:
            for delta in deltas:
                # Simulate awareness trajectory
                outcomes = self._simulate_outcomes(alpha, delta, n_simulations)
                
                results.append({
                    'alpha': alpha,
                    'delta': delta,
                    'final_awareness': outcomes['final_awareness'],
                    'scan_rate': outcomes['scan_rate'],
                    'awareness_variance': outcomes['awareness_variance'],
                    'convergence_time': outcomes['convergence_time']
                })
        
        result_df = pd.DataFrame(results)
        self.results['sensitivity'] = result_df
        return result_df
    
    def _simulate_outcomes(
        self, 
        alpha: float, 
        delta: float, 
        n_simulations: int
    ) -> Dict:
        # Simulate outcomes for given parameters
        np.random.seed(42)
        
        final_awareness_list = []
        scan_rates = []
        convergence_times = []
        
        for _ in range(n_simulations):
            # Simulate 7-day stay
            awareness = 0.0
            days = 7
            exposures_per_day = 2
            
            scans = 0
            impressions = 0
            
            for day in range(days):
                for _ in range(exposures_per_day):
                    # Exposure
                    awareness = awareness + alpha * (1 - awareness)
                    
                    # Scan probability
                    scan_prob = 0.3 * awareness
                    if np.random.random() < scan_prob:
                        scans += 1
                    impressions += 1
                
                # Decay overnight
                awareness = awareness * (1 - delta)
            
            final_awareness_list.append(awareness)
            scan_rates.append(scans / impressions if impressions > 0 else 0)
            
            # Estimate convergence time (when awareness reaches 90% of max)
            convergence_times.append(day + 1)
        
        return {
            'final_awareness': np.mean(final_awareness_list),
            'scan_rate': np.mean(scan_rates),
            'awareness_variance': np.var(final_awareness_list),
            'convergence_time': np.mean(convergence_times)
        }
    
    def noise_robustness(
        self,
        noise_levels: List[float] = [0.0, 0.01, 0.02, 0.05, 0.10],
        n_simulations: int = 100
    ) -> pd.DataFrame:
        # Test robustness to noise in awareness updates

        results = []
        
        for noise_level in noise_levels:
            np.random.seed(42)
            
            final_awareness_list = []
            
            for _ in range(n_simulations):
                awareness = 0.0
                
                for _ in range(14):  # 14 exposures
                    # Update with noise
                    noise = np.random.normal(0, noise_level)
                    awareness = awareness + self.base_alpha * (1 - awareness) + noise
                    awareness = np.clip(awareness, 0, 1)
                
                final_awareness_list.append(awareness)
            
            results.append({
                'noise_level': noise_level,
                'mean_awareness': np.mean(final_awareness_list),
                'std_awareness': np.std(final_awareness_list),
                'cv': np.std(final_awareness_list) / np.mean(final_awareness_list)
            })
        
        result_df = pd.DataFrame(results)
        self.results['noise_robustness'] = result_df
        return result_df
    
    def parameter_identifiability(
        self,
        true_alpha: float = 0.30,
        true_delta: float = 0.10,
        n_observations: List[int] = [50, 100, 200, 500, 1000]
    ) -> pd.DataFrame:
        # Test parameter identifiability: can we recover true α, δ from data?

        results = []
        
        for n_obs in n_observations:
            # Generate data with true parameters
            np.random.seed(42)
            awareness_trajectories = []
            
            for _ in range(n_obs):
                awareness = 0.0
                trajectory = [awareness]
                
                for _ in range(7):  # 7 exposures
                    awareness = awareness + true_alpha * (1 - awareness)
                    trajectory.append(awareness)
                    awareness = awareness * (1 - true_delta)
                    trajectory.append(awareness)
                
                awareness_trajectories.append(trajectory)
            
            # Estimate α from growth steps
            growth_changes = []
            for traj in awareness_trajectories:
                for i in range(0, len(traj) - 2, 2):
                    delta_rho = traj[i + 1] - traj[i]
                    one_minus_rho = 1 - traj[i]
                    if one_minus_rho > 0.01:
                        growth_changes.append(delta_rho / one_minus_rho)
            
            estimated_alpha = np.mean(growth_changes) if growth_changes else 0
            alpha_error = abs(estimated_alpha - true_alpha) / true_alpha
            
            results.append({
                'n_observations': n_obs,
                'true_alpha': true_alpha,
                'estimated_alpha': estimated_alpha,
                'alpha_error': alpha_error,
                'identifiable': alpha_error < 0.1
            })
        
        result_df = pd.DataFrame(results)
        self.results['identifiability'] = result_df
        return result_df


# Fairness analysis

class FairnessAnalyzer:
    # Counterfactual fairness analysis for ad exposure.
    
    def __init__(self, impressions: pd.DataFrame):
        self.impressions = impressions
        self.results = {}
    
    def segment_exposure_balance(self) -> pd.DataFrame:
        # Measure balance of exposure across segments.

        if 'segment_id' not in self.impressions.columns:
            return pd.DataFrame()
        
        segment_exposure = self.impressions.groupby('segment_id').size()
        
        # Expected (uniform) distribution
        expected = len(self.impressions) / len(segment_exposure)
        
        # Actual proportions
        proportions = segment_exposure / segment_exposure.sum()
        
        # Gini coefficient
        sorted_props = np.sort(proportions.values)
        n = len(sorted_props)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_props)) - (n + 1)) / n
        
        result = pd.DataFrame({
            'segment': segment_exposure.index,
            'actual_exposure': segment_exposure.values,
            'expected_exposure': [expected] * len(segment_exposure),
            'ratio': segment_exposure.values / expected
        })
        
        self.results['segment_balance'] = {
            'distribution': result,
            'gini_coefficient': gini,
            'interpretation': 'Fair' if gini < 0.3 else 'Unfair'
        }
        
        return result
    
    def category_exposure_fairness(self) -> pd.DataFrame:
        # Measure fairness of category exposure across segments.

        if 'segment_id' not in self.impressions.columns or 'category' not in self.impressions.columns:
            return pd.DataFrame()
        
        # Cross-tabulation
        cross_tab = pd.crosstab(
            self.impressions['segment_id'],
            self.impressions['category'],
            normalize='index'
        )
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = stats.chi2_contingency(
            pd.crosstab(self.impressions['segment_id'], self.impressions['category'])
        )
        
        self.results['category_fairness'] = {
            'distribution': cross_tab,
            'chi2': chi2,
            'p_value': p_value,
            'interpretation': 'Fair (independent)' if p_value > 0.05 else 'Unfair (dependent)'
        }
        
        return cross_tab
    
    def advertiser_fairness(self) -> Dict:
        # Measure fairness from advertiser perspective.

        if 'advertiser_id' not in self.impressions.columns:
            return {}
        
        advertiser_exposure = self.impressions.groupby('advertiser_id').size()
        
        n = len(advertiser_exposure)
        sum_x = advertiser_exposure.sum()
        sum_x2 = (advertiser_exposure ** 2).sum()
        
        jains_index = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1
        
        # Gini coefficient
        sorted_exp = np.sort(advertiser_exposure.values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_exp)) - (n + 1) * sum_x) / (n * sum_x) if sum_x > 0 else 0
        
        self.results['advertiser_fairness'] = {
            'jains_index': jains_index,
            'gini_coefficient': gini,
            'interpretation': 'Fair' if jains_index > 0.7 else 'Unfair',
            'n_advertisers': n,
            'exposure_range': (advertiser_exposure.min(), advertiser_exposure.max()),
            'exposure_std': advertiser_exposure.std()
        }
        
        return self.results['advertiser_fairness']


# Main execution

def run_causal_analysis_demo():
  # Generate synthetic data for demonstration
    np.random.seed(42)
    n_impressions = 5000
    
    data = pd.DataFrame({
        'guest_id': np.random.randint(0, 500, n_impressions),
        'segment_id': np.random.randint(0, 8, n_impressions),
        'advertiser_id': np.random.randint(0, 100, n_impressions),
        'category': np.random.choice(['Restaurants', 'Experiences', 'Shopping', 'Wellness', 'Nightlife'], n_impressions),
        'exposed': np.random.binomial(1, 0.7, n_impressions),
        'awareness': np.random.uniform(0.1, 0.8, n_impressions),
        'scanned': np.random.binomial(1, 0.15, n_impressions),
        'hour': np.random.randint(6, 24, n_impressions),
        'day_of_stay': np.random.randint(1, 8, n_impressions)
    })
    
    # 1. Popularity Baseline
    baseline = PopularityBaseline(data)
    impression_pop = baseline.compute_impression_popularity()
    engagement_pop = baseline.compute_engagement_popularity()
    print("Impression Popularity:")
    print(impression_pop)
    print("\nEngagement Popularity:")
    print(engagement_pop)
    
    # 2. Endogeneity Analysis
    endogeneity = EndogeneityAnalyzer(data)
    randomness = endogeneity.test_exposure_randomness()
    instruments = endogeneity.identify_instruments()
    comparison = endogeneity.compare_endogeneity_sources()
    print("Randomness Test:", randomness)
    print("Potential Instruments:", instruments)
    print("\nEndogeneity Comparison (Web vs In-Room TV):")
    print(comparison.to_string())
    
    # 3. Causal Effects
    causal = CausalEffectEstimator(data)
    ate_naive = causal.estimate_ATE_naive()
    propensity = causal.estimate_propensity_scores()
    ate_ipw = causal.estimate_ATE_IPW(propensity_scores=propensity)
    
    print(f"Naive ATE: {ate_naive['ATE']:.4f} [{ate_naive['CI_lower']:.4f}, {ate_naive['CI_upper']:.4f}]")
    print(f"IPW ATE:   {ate_ipw['ATE']:.4f} [{ate_ipw['CI_lower']:.4f}, {ate_ipw['CI_upper']:.4f}]")
    
    awareness_effect = causal.estimate_awareness_causal_effect()
    print(f"\nAwareness Effect: {awareness_effect['interpretation']}")
    
    # 4. Robustness Analysis
    robustness = RobustnessAnalyzer()
    sensitivity = robustness.sensitivity_analysis(n_steps=5, n_simulations=50)
    noise_robust = robustness.noise_robustness(n_simulations=50)
    identifiability = robustness.parameter_identifiability()
    
    print("Sensitivity Analysis (sample):")
    print(sensitivity.head())
    print("\nNoise Robustness:")
    print(noise_robust)
    print("\nParameter Identifiability:")
    print(identifiability)
    
    # 5. Fairness Analysis
    fairness = FairnessAnalyzer(data)
    segment_balance = fairness.segment_exposure_balance()
    advertiser_fair = fairness.advertiser_fairness()
    
    print("Segment Balance:")
    print(segment_balance)
    print(f"\nSegment Gini: {fairness.results['segment_balance']['gini_coefficient']:.3f}")
    print(f"Advertiser Jain's Index: {advertiser_fair['jains_index']:.3f}")
    
    return {
        'baseline': baseline,
        'endogeneity': endogeneity,
        'causal': causal,
        'robustness': robustness,
        'fairness': fairness
    }


if __name__ == "__main__":
    results = run_causal_analysis_demo()




