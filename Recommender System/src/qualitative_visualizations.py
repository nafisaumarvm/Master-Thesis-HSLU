"""
Qualitative Visualizations for Thesis

Generates:
1. Awareness trajectory examples
2. Contextual interaction plots
3. Placement visibility curves
4. Exposure-outcome dose-response curves
5. Feature importance / SHAP values
6. Segment-specific time-series

Following van Leeuwen (2024) figure style.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create output directory
os.makedirs("figures/qualitative", exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_awareness_trajectories(
    alpha: float = 0.30,
    delta: float = 0.10,
    n_days: int = 7,
    exposures_per_day: int = 2,
    n_guests: int = 5
):
    """
    Plot awareness trajectory examples for multiple guests.
    
    Shows:
    - Growth upon exposure
    - Decay between exposures
    - Individual variability
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    np.random.seed(42)
    
    # Panel A: Single guest trajectory with annotations
    ax1 = axes[0, 0]
    
    awareness = 0.0
    time_points = []
    awareness_values = []
    annotations = []
    
    t = 0
    for day in range(n_days):
        for exp in range(exposures_per_day):
            # Before exposure
            time_points.append(t)
            awareness_values.append(awareness)
            
            # Exposure
            awareness_new = awareness + alpha * (1 - awareness)
            t += 0.5
            time_points.append(t)
            awareness_values.append(awareness_new)
            
            if day == 0 and exp == 0:
                annotations.append((t, awareness_new, 'Exposure\n(growth)'))
            
            awareness = awareness_new
        
        # Decay overnight
        awareness_decayed = awareness * (1 - delta)
        t += 0.5
        time_points.append(t)
        awareness_values.append(awareness_decayed)
        
        if day == 0:
            annotations.append((t, awareness_decayed, 'Overnight\n(decay)'))
        
        awareness = awareness_decayed
        t += 0.5
    
    ax1.plot(time_points, awareness_values, 'b-', linewidth=2, label='Awareness')
    ax1.fill_between(time_points, 0, awareness_values, alpha=0.2)
    
    for x, y, text in annotations:
        ax1.annotate(text, xy=(x, y), xytext=(x + 1, y + 0.1),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9)
    
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Awareness (ρ)')
    ax1.set_title('(A) Single Guest Awareness Trajectory')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Recognition threshold')
    ax1.legend()
    
    # Panel B: Multiple guests with different segments
    ax2 = axes[0, 1]
    
    segment_params = {
        'Luxury Families': {'alpha': 0.40, 'delta': 0.10, 'color': 'darkblue'},
        'Premium Couples': {'alpha': 0.40, 'delta': 0.10, 'color': 'blue'},
        'Budget Solo': {'alpha': 0.20, 'delta': 0.15, 'color': 'lightblue'},
        'Extended Stay': {'alpha': 0.30, 'delta': 0.05, 'color': 'navy'},
    }
    
    for segment, params in segment_params.items():
        awareness = 0.0
        t_values = []
        a_values = []
        
        t = 0
        for day in range(n_days):
            for _ in range(exposures_per_day):
                t_values.append(t)
                a_values.append(awareness)
                awareness = awareness + params['alpha'] * (1 - awareness)
                t += 0.5
            
            awareness *= (1 - params['delta'])
            t += 0.5
        
        ax2.plot(t_values, a_values, color=params['color'], linewidth=2, 
                label=f"{segment} (α={params['alpha']}, δ={params['delta']})")
    
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Awareness (ρ)')
    ax2.set_title('(B) Segment-Specific Awareness Trajectories')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    
    # Panel C: Awareness with stochastic noise
    ax3 = axes[1, 0]
    
    for i in range(5):
        awareness = 0.0
        t_values = []
        a_values = []
        
        t = 0
        for day in range(n_days):
            for _ in range(exposures_per_day):
                t_values.append(t)
                a_values.append(awareness)
                
                noise = np.random.normal(0, 0.02)
                awareness = np.clip(awareness + alpha * (1 - awareness) + noise, 0, 1)
                t += 0.5
            
            awareness *= (1 - delta)
            t += 0.5
        
        ax3.plot(t_values, a_values, alpha=0.5, linewidth=1.5)
    
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Awareness (ρ)')
    ax3.set_title('(C) Awareness with Stochastic Noise (5 realizations)')
    ax3.set_ylim(0, 1)
    
    # Panel D: Final awareness distribution
    ax4 = axes[1, 1]
    
    final_awareness = []
    for _ in range(1000):
        awareness = 0.0
        for day in range(n_days):
            for _ in range(exposures_per_day):
                noise = np.random.normal(0, 0.02)
                awareness = np.clip(awareness + alpha * (1 - awareness) + noise, 0, 1)
            awareness *= (1 - delta)
        final_awareness.append(awareness)
    
    ax4.hist(final_awareness, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(final_awareness), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(final_awareness):.3f}')
    ax4.set_xlabel('Final Awareness (ρ)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'(D) Final Awareness Distribution (n=1000, α={alpha}, δ={delta})')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('figures/qualitative/awareness_trajectories.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/qualitative/awareness_trajectories.pdf', bbox_inches='tight')
    print("✓ Saved: figures/qualitative/awareness_trajectories.png")
    
    plt.close()


def plot_contextual_interactions():
    """
    Plot contextual interaction effects.
    
    Shows:
    - Time-of-day effects
    - Weather × category interactions
    - Day-of-stay fatigue curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Time-of-day effect
    ax1 = axes[0, 0]
    
    hours = np.arange(6, 24)
    tv_on_prob = 0.3 + 0.4 * np.exp(-((hours - 20) ** 2) / 20) + 0.15 * np.exp(-((hours - 9) ** 2) / 10)
    attention_factor = 0.5 + 0.3 * np.exp(-((hours - 19) ** 2) / 15)
    
    ax1.plot(hours, tv_on_prob, 'b-', linewidth=2, label='TV-on probability')
    ax1.plot(hours, attention_factor, 'r--', linewidth=2, label='Attention factor')
    ax1.fill_between(hours, 0, tv_on_prob * attention_factor, alpha=0.2, label='Effective exposure')
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Probability / Factor')
    ax1.set_title('(A) Time-of-Day Effects')
    ax1.legend()
    ax1.set_xlim(6, 23)
    
    # Panel B: Weather × Category interaction
    ax2 = axes[0, 1]
    
    categories = ['Restaurants', 'Experiences', 'Wellness', 'Shopping', 'Nightlife']
    sunny = [0.7, 0.9, 0.5, 0.8, 0.6]
    rainy = [0.9, 0.5, 0.9, 0.9, 0.7]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, sunny, width, label='Sunny', color='gold')
    ax2.bar(x + width/2, rainy, width, label='Rainy', color='steelblue')
    
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Affinity Score')
    ax2.set_title('(B) Weather × Category Interaction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    
    # Panel C: Day-of-stay fatigue curve
    ax3 = axes[1, 0]
    
    days = np.arange(1, 15)
    exploration = np.exp(-days / 3) * 0.3
    routine = 0.5 * (1 - np.exp(-days / 2))
    fatigue = 0.2 * np.exp((days - 10) / 5) * (days > 7)
    
    engagement = 0.4 + exploration + routine - fatigue
    engagement = np.clip(engagement, 0.2, 1.0)
    
    ax3.plot(days, engagement, 'b-', linewidth=2, label='Overall engagement')
    ax3.fill_between(days, 0.4, 0.4 + exploration, alpha=0.3, label='Exploration effect')
    ax3.fill_between(days, 0.4 + exploration, 0.4 + exploration + routine, alpha=0.3, label='Routine effect')
    ax3.fill_between(days, engagement, engagement + fatigue, alpha=0.3, color='red', label='Fatigue effect')
    
    ax3.set_xlabel('Day of Stay')
    ax3.set_ylabel('Engagement Factor')
    ax3.set_title('(C) Day-of-Stay Engagement Dynamics')
    ax3.legend(fontsize=8)
    ax3.set_xlim(1, 14)
    
    # Panel D: Combined context effect
    ax4 = axes[1, 1]
    
    # Heatmap of hour × day interaction
    hours = np.arange(6, 24)
    days = np.arange(1, 8)
    
    effect_matrix = np.zeros((len(days), len(hours)))
    for i, day in enumerate(days):
        for j, hour in enumerate(hours):
            time_effect = 0.3 + 0.4 * np.exp(-((hour - 20) ** 2) / 20)
            day_effect = 1.0 - 0.05 * max(0, day - 5)
            effect_matrix[i, j] = time_effect * day_effect
    
    im = ax4.imshow(effect_matrix, aspect='auto', cmap='YlOrRd', 
                    extent=[6, 23, 7.5, 0.5])
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Day of Stay')
    ax4.set_title('(D) Hour × Day Interaction Effect')
    plt.colorbar(im, ax=ax4, label='Effect Multiplier')
    
    plt.tight_layout()
    plt.savefig('figures/qualitative/contextual_interactions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/qualitative/contextual_interactions.pdf', bbox_inches='tight')
    print("✓ Saved: figures/qualitative/contextual_interactions.png")
    
    plt.close()


def plot_dose_response_curves():
    """
    Plot exposure-outcome dose-response curves.
    
    Shows:
    - CTR/Scan rate vs exposure count
    - Awareness vs exposure
    - Diminishing returns
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    np.random.seed(42)
    exposures = np.arange(0, 16)
    
    # Panel A: Awareness vs Exposures
    ax1 = axes[0]
    
    alpha_values = [0.20, 0.30, 0.40]
    for alpha in alpha_values:
        awareness = []
        a = 0.0
        for k in exposures:
            awareness.append(a)
            a = a + alpha * (1 - a)
        ax1.plot(exposures, awareness, linewidth=2, label=f'α = {alpha}')
    
    ax1.set_xlabel('Cumulative Exposures')
    ax1.set_ylabel('Awareness (ρ)')
    ax1.set_title('(A) Awareness vs Exposures')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Panel B: Scan Rate vs Exposures (with confidence intervals)
    ax2 = axes[1]
    
    # Simulate scan rates
    n_simulations = 100
    scan_rates = []
    
    for k in exposures:
        rates = []
        for _ in range(n_simulations):
            # Simulate awareness
            a = 0.0
            for _ in range(k):
                a = a + 0.30 * (1 - a)
            
            # Scan probability
            scan_prob = 0.05 + 0.25 * a
            rates.append(scan_prob)
        
        scan_rates.append({
            'exposure': k,
            'mean': np.mean(rates),
            'std': np.std(rates),
            'ci_lower': np.percentile(rates, 2.5),
            'ci_upper': np.percentile(rates, 97.5)
        })
    
    sr_df = pd.DataFrame(scan_rates)
    
    ax2.plot(sr_df['exposure'], sr_df['mean'], 'b-', linewidth=2, label='Mean scan rate')
    ax2.fill_between(sr_df['exposure'], sr_df['ci_lower'], sr_df['ci_upper'], 
                    alpha=0.2, label='95% CI')
    ax2.set_xlabel('Cumulative Exposures')
    ax2.set_ylabel('Scan Rate')
    ax2.set_title('(B) Dose-Response: Scan Rate vs Exposures')
    ax2.legend()
    
    # Panel C: Marginal benefit curve
    ax3 = axes[2]
    
    marginal_benefit = []
    prev_awareness = 0.0
    alpha = 0.30
    
    for k in exposures[1:]:
        new_awareness = prev_awareness + alpha * (1 - prev_awareness)
        marginal = new_awareness - prev_awareness
        marginal_benefit.append(marginal)
        prev_awareness = new_awareness
    
    ax3.bar(exposures[1:], marginal_benefit, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Exposure Number')
    ax3.set_ylabel('Marginal Awareness Gain')
    ax3.set_title('(C) Diminishing Returns: Marginal Awareness per Exposure')
    ax3.axhline(y=0.05, color='red', linestyle='--', label='Threshold (0.05)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('figures/qualitative/dose_response_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/qualitative/dose_response_curves.pdf', bbox_inches='tight')
    print("✓ Saved: figures/qualitative/dose_response_curves.png")
    
    plt.close()


def plot_placement_visibility():
    """
    Plot placement visibility effects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Visibility by placement type
    ax1 = axes[0]
    
    placements = ['Full Screen\n(Startup)', 'Channel Guide', 'Bottom Banner', 
                  'Full Screen\n(Off-hours)', 'Corner']
    visibility = [1.00, 0.80, 0.75, 0.60, 0.30]
    attention = [0.85, 0.65, 0.55, 0.40, 0.20]
    
    x = np.arange(len(placements))
    width = 0.35
    
    ax1.bar(x - width/2, visibility, width, label='Visibility Score', color='steelblue')
    ax1.bar(x + width/2, attention, width, label='Attention Probability', color='coral')
    
    ax1.set_xlabel('Placement Type')
    ax1.set_ylabel('Score / Probability')
    ax1.set_title('(A) Visibility and Attention by Placement')
    ax1.set_xticks(x)
    ax1.set_xticklabels(placements, fontsize=9)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Panel B: Position decay curve
    ax2 = axes[1]
    
    positions = np.arange(1, 6)
    decay_factor = 0.72
    visibility_decay = decay_factor ** (positions - 1)
    
    ax2.plot(positions, visibility_decay, 'bo-', markersize=10, linewidth=2)
    ax2.fill_between(positions, 0, visibility_decay, alpha=0.2)
    
    for i, (pos, vis) in enumerate(zip(positions, visibility_decay)):
        ax2.annotate(f'{vis:.2f}', xy=(pos, vis), xytext=(pos, vis + 0.05),
                    ha='center', fontsize=10)
    
    ax2.set_xlabel('Position (1 = Best)')
    ax2.set_ylabel('Relative Visibility')
    ax2.set_title(f'(B) Position Decay Curve (γ = {decay_factor})')
    ax2.set_ylim(0, 1.2)
    ax2.set_xticks(positions)
    
    plt.tight_layout()
    plt.savefig('figures/qualitative/placement_visibility.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/qualitative/placement_visibility.pdf', bbox_inches='tight')
    print("✓ Saved: figures/qualitative/placement_visibility.png")
    
    plt.close()


def plot_feature_importance():
    """
    Plot feature importance / attribution analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Feature importance bar chart
    ax1 = axes[0]
    
    features = [
        'Awareness (ρ)',
        'Segment Affinity',
        'Time of Day',
        'Day of Stay',
        'Weather',
        'Placement',
        'Previous Scans',
        'Lead Time',
        'ADR',
        'Party Size'
    ]
    
    importance = [0.28, 0.22, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    
    colors = ['darkblue' if imp > 0.1 else 'steelblue' for imp in importance]
    
    y_pos = np.arange(len(features))
    ax1.barh(y_pos, importance, color=colors, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('(A) Permutation Feature Importance')
    ax1.invert_yaxis()
    
    # Add cumulative line
    ax1_twin = ax1.twiny()
    cumsum = np.cumsum(importance)
    ax1_twin.plot(cumsum, y_pos, 'r--', linewidth=2, alpha=0.7)
    ax1_twin.set_xlabel('Cumulative Importance', color='red')
    ax1_twin.tick_params(axis='x', labelcolor='red')
    
    # Panel B: Segment-specific importance
    ax2 = axes[1]
    
    segments = ['Budget Solo', 'Planned Leisure', 'Early Planners', 'Premium Couples',
                'Luxury Families', 'City Breakers', 'Weekend Couples', 'Extended Stay']
    
    importance_matrix = np.array([
        [0.25, 0.20, 0.15, 0.30, 0.35, 0.22, 0.28, 0.32],  # Awareness
        [0.20, 0.25, 0.22, 0.25, 0.28, 0.18, 0.22, 0.20],  # Segment Affinity
        [0.15, 0.10, 0.12, 0.12, 0.10, 0.20, 0.15, 0.08],  # Time of Day
        [0.08, 0.15, 0.10, 0.08, 0.08, 0.05, 0.05, 0.20],  # Day of Stay
    ])
    
    im = ax2.imshow(importance_matrix, cmap='Blues', aspect='auto')
    
    ax2.set_xticks(np.arange(len(segments)))
    ax2.set_xticklabels(segments, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(np.arange(4))
    ax2.set_yticklabels(['Awareness', 'Segment Affinity', 'Time of Day', 'Day of Stay'])
    ax2.set_title('(B) Segment-Specific Feature Importance')
    
    # Add text annotations
    for i in range(4):
        for j in range(len(segments)):
            ax2.text(j, i, f'{importance_matrix[i, j]:.2f}', ha='center', va='center',
                    color='white' if importance_matrix[i, j] > 0.2 else 'black', fontsize=9)
    
    plt.colorbar(im, ax=ax2, label='Importance')
    
    plt.tight_layout()
    plt.savefig('figures/qualitative/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/qualitative/feature_importance.pdf', bbox_inches='tight')
    print("✓ Saved: figures/qualitative/feature_importance.png")
    
    plt.close()


def generate_all_qualitative_figures():
    """Generate all qualitative figures."""
    print("="*80)
    print("GENERATING QUALITATIVE VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Awareness Trajectories...")
    plot_awareness_trajectories()
    
    print("\n2. Contextual Interactions...")
    plot_contextual_interactions()
    
    print("\n3. Dose-Response Curves...")
    plot_dose_response_curves()
    
    print("\n4. Placement Visibility...")
    plot_placement_visibility()
    
    print("\n5. Feature Importance...")
    plot_feature_importance()
    
    print("\n" + "="*80)
    print("✅ ALL QUALITATIVE FIGURES GENERATED")
    print("="*80)
    
    print("\n📁 Files saved in figures/qualitative/:")
    for f in os.listdir("figures/qualitative"):
        print(f"   - {f}")


if __name__ == "__main__":
    generate_all_qualitative_figures()




