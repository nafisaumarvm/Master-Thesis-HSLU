"""
Utility functions for the contextual ad recommender system.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy.special import expit as sigmoid


def set_random_seed(seed: int = 42) -> np.random.Generator:
    """
    Create a seeded numpy random generator for reproducibility.
    
    Args:
        seed: Random seed
        
    Returns:
        numpy.random.Generator instance
    """
    return np.random.default_rng(seed)


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute logit (inverse sigmoid) of probability p.
    
    Args:
        p: Probability value(s) in [0, 1]
        
    Returns:
        Logit value(s)
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def safe_sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute sigmoid with numerical stability.
    
    Args:
        x: Input value(s)
        
    Returns:
        Sigmoid value(s) in [0, 1]
    """
    return sigmoid(x)


def normalize_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert scores to probabilities using softmax.
    
    Args:
        scores: Array of scores
        temperature: Temperature parameter (higher = more uniform)
        
    Returns:
        Normalized probabilities
    """
    scores = np.asarray(scores)
    exp_scores = np.exp((scores - scores.max()) / temperature)
    return exp_scores / exp_scores.sum()


def sample_from_distribution(
    values: List[Any],
    probabilities: Optional[np.ndarray] = None,
    size: int = 1,
    rng: Optional[np.random.Generator] = None
) -> Union[Any, List[Any]]:
    """
    Sample from a discrete distribution.
    
    Args:
        values: List of values to sample from
        probabilities: Probability of each value (uniform if None)
        size: Number of samples
        rng: Random generator
        
    Returns:
        Single value if size=1, else list of values
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        probabilities = probabilities / probabilities.sum()
    
    indices = rng.choice(len(values), size=size, p=probabilities, replace=True)
    samples = [values[i] for i in indices]
    
    return samples[0] if size == 1 else samples


def compute_empirical_distribution(
    series: pd.Series,
    normalize: bool = True
) -> Dict[Any, float]:
    """
    Compute empirical distribution from a pandas Series.
    
    Args:
        series: Input series
        normalize: If True, return probabilities; else counts
        
    Returns:
        Dictionary mapping values to probabilities/counts
    """
    counts = series.value_counts()
    if normalize:
        counts = counts / counts.sum()
    return counts.to_dict()


def jitter_dates(
    dates: pd.Series,
    max_days: int = 365,
    rng: Optional[np.random.Generator] = None
) -> pd.Series:
    """
    Add random jitter to dates for privacy.
    
    Args:
        dates: Series of datetime objects
        max_days: Maximum number of days to jitter
        rng: Random generator
        
    Returns:
        Jittered dates
    """
    if rng is None:
        rng = np.random.default_rng()
    
    jitter = rng.integers(-max_days, max_days, size=len(dates))
    return dates + pd.to_timedelta(jitter, unit='D')


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode categorical columns.
    
    Args:
        df: Input dataframe
        columns: Columns to encode
        method: 'onehot' or 'label'
        
    Returns:
        Dataframe with encoded columns
    """
    df = df.copy()
    
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df


def clip_probability(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Clip probability to valid range [0, 1].
    
    Args:
        p: Probability value(s)
        
    Returns:
        Clipped probability
    """
    return np.clip(p, 0.0, 1.0)


def generate_fake_ids(
    n: int,
    prefix: str = "ID",
    rng: Optional[np.random.Generator] = None
) -> List[str]:
    """
    Generate fake unique IDs.
    
    Args:
        n: Number of IDs
        prefix: ID prefix
        rng: Random generator
        
    Returns:
        List of unique IDs
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate unique random numbers
    random_nums = rng.choice(10**10, size=n, replace=False)
    return [f"{prefix}_{num:010d}" for num in random_nums]


def parse_tags_string(tags: str) -> List[str]:
    """
    Parse comma-separated tag string into list.
    
    Args:
        tags: String like "tag1,tag2,tag3"
        
    Returns:
        List of tags
    """
    if pd.isna(tags) or tags == "":
        return []
    return [t.strip() for t in str(tags).split(',')]


def tags_to_string(tags: List[str]) -> str:
    """
    Convert list of tags to comma-separated string.
    
    Args:
        tags: List of tags
        
    Returns:
        Comma-separated string
    """
    return ','.join(tags)





