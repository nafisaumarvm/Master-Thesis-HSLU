# Initial Evaluation metrics and counterfactual estimation.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.metrics import ndcg_score
import warnings


def compute_ctr(exposure_log: pd.DataFrame) -> float:
    # Compute Click-Through Rate

    if len(exposure_log) == 0:
        return 0.0
    
    return exposure_log['click'].mean()


def compute_rpm(exposure_log: pd.DataFrame) -> float:
    # Compute Revenue Per Mille (per 1000 impressions)

    if len(exposure_log) == 0:
        return 0.0
    
    total_revenue = exposure_log['revenue'].sum()
    n_impressions = len(exposure_log)
    
    rpm = (total_revenue / n_impressions) * 1000
    
    return rpm


def compute_revenue_per_stay(
    exposure_log: pd.DataFrame,
    n_stays: Optional[int] = None
) -> float:
    # Compute average revenue per stay

    total_revenue = exposure_log['revenue'].sum()
    
    if n_stays is None:
        n_stays = exposure_log['stay_id'].nunique()
    
    if n_stays == 0:
        return 0.0
    
    return total_revenue / n_stays


def compute_guest_experience_index(
    exposure_log: pd.DataFrame,
    intrusion_col: str = 'intrusion_cost'
) -> float:
    # Compute guest experience index (negative intrusion)   

    if intrusion_col not in exposure_log.columns:
        return 0.0
    
    avg_intrusion = exposure_log[intrusion_col].mean()
    
    # Negative intrusion as index (higher = better)
    return -avg_intrusion


def precision_at_k(
    recommended_ads: List[str],
    relevant_ads: List[str],
    k: int
) -> float:
    # Compute Precision@k

    if k == 0 or len(recommended_ads) == 0:
        return 0.0
    
    top_k = recommended_ads[:k]
    relevant_set = set(relevant_ads)
    
    n_relevant = sum(1 for ad in top_k if ad in relevant_set)
    
    return n_relevant / k


def recall_at_k(
    recommended_ads: List[str],
    relevant_ads: List[str],
    k: int
) -> float:
    # Compute Recall@k
    if len(relevant_ads) == 0:
        return 0.0
    
    top_k = recommended_ads[:k]
    relevant_set = set(relevant_ads)
    
    n_relevant = sum(1 for ad in top_k if ad in relevant_set)
    
    return n_relevant / len(relevant_ads)


def ndcg_at_k(
    recommended_ads: List[str],
    ad_scores: Dict[str, float],
    k: int
) -> float:
    # Compute NDCG@k
    if k == 0 or len(recommended_ads) == 0:
        return 0.0
    
    # Get scores for recommended ads
    top_k = recommended_ads[:k]
    scores = [ad_scores.get(ad, 0.0) for ad in top_k]
    
    # Pad if necessary
    while len(scores) < k:
        scores.append(0.0)
    
    # Ideal ranking (sorted by score)
    ideal_scores = sorted(ad_scores.values(), reverse=True)[:k]
    while len(ideal_scores) < k:
        ideal_scores.append(0.0)
    
    # Use sklearn's ndcg_score
    try:
        ndcg = ndcg_score([ideal_scores], [scores])
    except:
        # Fallback to manual computation
        def dcg(scores):
            return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
        
        dcg_val = dcg(scores)
        idcg_val = dcg(ideal_scores)
        
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
    
    return ndcg


def evaluate_ranking(
    model,
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    guest_ad_prefs_df: pd.DataFrame,
    k: int = 3,
    n_samples: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    # Evaluate ranking quality using ground-truth preferences

    rng = np.random.default_rng(seed)
    
    # Sample guests
    if len(guests_df) > n_samples:
        sample_indices = rng.choice(len(guests_df), size=n_samples, replace=False)
        sample_guests = guests_df.iloc[sample_indices]
    else:
        sample_guests = guests_df
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for _, guest in sample_guests.iterrows():
        guest_id = guest['guest_id']
        
        # Get ground-truth relevant ads (top by intrinsic_score)
        guest_prefs = guest_ad_prefs_df[
            guest_ad_prefs_df['guest_id'] == guest_id
        ].sort_values('intrinsic_score', ascending=False)
        
        if len(guest_prefs) == 0:
            continue
        
        # Get candidate ads
        candidate_ad_ids = guest_prefs['ad_id'].tolist()
        candidate_ads = ads_df[ads_df['ad_id'].isin(candidate_ad_ids)]
        
        if len(candidate_ads) == 0:
            continue
        
        # Model predictions
        try:
            if hasattr(model, 'rank'):
                recommended = model.rank(guest, candidate_ads, k=k)
            elif hasattr(model, 'predict_proba'):
                probs = model.predict_proba(guest, candidate_ads)
                top_k_indices = np.argsort(probs)[::-1][:k]
                recommended = candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
            else:
                # Fallback: random
                recommended = candidate_ads.head(k)['ad_id'].tolist()
        except:
            continue
        
        # Ground truth: top-k by intrinsic_score
        relevant_ads = guest_prefs.head(k)['ad_id'].tolist()
        
        # Compute metrics
        p = precision_at_k(recommended, relevant_ads, k)
        r = recall_at_k(recommended, relevant_ads, k)
        
        # NDCG
        ad_scores = guest_prefs.set_index('ad_id')['intrinsic_score'].to_dict()
        n = ndcg_at_k(recommended, ad_scores, k)
        
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)
    
    metrics = {
        'precision@k': np.mean(precisions) if precisions else 0.0,
        'recall@k': np.mean(recalls) if recalls else 0.0,
        'ndcg@k': np.mean(ndcgs) if ndcgs else 0.0,
        'n_evaluated': len(precisions)
    }
    
    return metrics


def inverse_propensity_score(
    exposure_log: pd.DataFrame,
    target_policy_fn: Callable,
    reward_col: str = 'click'
) -> float:
    # Compute IPS estimate of expected reward for a target policy

    if len(exposure_log) == 0:
        return 0.0
    
    ips_scores = []
    
    for _, row in exposure_log.iterrows():
        guest_id = row['guest_id']
        ad_id = row['ad_id']
        reward = row[reward_col]
        propensity = row['propensity']
        
        # Target policy probability
        target_prob = target_policy_fn(guest_id, ad_id)
        
        # IPS weight
        if propensity > 0:
            weight = target_prob / propensity
        else:
            weight = 0.0
        
        ips_scores.append(weight * reward)
    
    return np.mean(ips_scores)


def doubly_robust_estimator(
    exposure_log: pd.DataFrame,
    target_policy_fn: Callable,
    outcome_model,
    reward_col: str = 'click'
) -> float:
    # Doubly-robust estimator for counterfactual evaluation

    if len(exposure_log) == 0:
        return 0.0
    
    dr_scores = []
    
    for _, row in exposure_log.iterrows():
        guest_id = row['guest_id']
        ad_id = row['ad_id']
        reward = row[reward_col]
        propensity = row['propensity']
        
        # Target policy probability
        target_prob = target_policy_fn(guest_id, ad_id)
        
        # Predicted reward from outcome model
        try:
            r_hat = reward  # Simplified
        except:
            r_hat = 0.0
        
        # IPS weight
        if propensity > 0:
            weight = target_prob / propensity
        else:
            weight = 0.0
        
        # DR term
        dr_term = weight * (reward - r_hat) + r_hat
        dr_scores.append(dr_term)
    
    return np.mean(dr_scores)


def evaluate_exposure_log(exposure_log: pd.DataFrame) -> Dict[str, float]:
    # Compute standard metrics from exposure log

    metrics = {
        'ctr': compute_ctr(exposure_log),
        'rpm': compute_rpm(exposure_log),
        'revenue_per_stay': compute_revenue_per_stay(exposure_log),
        'total_revenue': exposure_log['revenue'].sum() if 'revenue' in exposure_log.columns else 0,
        'total_clicks': exposure_log['click'].sum() if 'click' in exposure_log.columns else 0,
        'total_impressions': len(exposure_log)
    }
    
    # Guest experience if available
    if 'intrusion_cost' in exposure_log.columns:
        metrics['guest_experience_index'] = compute_guest_experience_index(exposure_log)
        metrics['avg_intrusion'] = exposure_log['intrusion_cost'].mean()
    
    return metrics


def compare_models(
    models: Dict[str, any],
    exposure_log: pd.DataFrame,
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    guest_ad_prefs_df: pd.DataFrame,
    k: int = 3
) -> pd.DataFrame:
    # Compare multiple models on ranking metrics

    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating: {model_name}")
        
        # Fit model if not already fitted
        if hasattr(model, 'fit') and not hasattr(model, 'ad_ctr'):
            try:
                model.fit(exposure_log, guests_df, ads_df)
            except:
                try:
                    model.fit(exposure_log)
                except:
                    pass
        
        # Evaluate ranking
        ranking_metrics = evaluate_ranking(
            model, guests_df, ads_df, guest_ad_prefs_df, k=k
        )
        
        result = {
            'model': model_name,
            **ranking_metrics
        }
        
        results.append(result)
    
    return pd.DataFrame(results)





