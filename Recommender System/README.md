# In-Room TV Advertising Recommender System

**Master Thesis Project**  
**Context:** Awareness-driven in-room TV advertising for hotel guests  
**Methodology:** Following van Leeuwen (2024) with extensions

---

## Overview

This system implements a complete pipeline for **in-room TV advertising** that optimizes for **visibility, reach, and awareness building** (not click-through rates). Unlike web advertising (CTR: 5-8%), in-room TV ads prioritize passive exposure with optional QR code engagement (scan rates: 0.5-2%).

### System Architecture

The recommender system is built on a multi-layered architecture that processes hotel booking data (119K bookings) and real Swiss advertiser data (801 establishments) to create personalized ad recommendations. The system first performs data-driven guest segmentation using hierarchical clustering and k-means to identify 8 distinct guest segments with unique preferences and behaviors. Each segment is characterized by specific awareness dynamics (learning rates α: 0.15-0.50, decay rates δ: 0.02-0.20) that model how guests build awareness of advertised establishments over their stay. The recommendation engine combines baseline machine learning models (Logistic Regression, XGBoost) with reinforcement learning policies (ε-greedy) to optimize ad selection. The simulation framework tracks awareness growth, exposure frequency, and click probabilities while enforcing guest experience constraints (1-2 ads per day, content filtering, no interruptions). The system evaluates performance across multiple objectives including reach (82.4%), awareness uplift (0.287), revenue per guest (€6.44), and guest experience metrics, providing a comprehensive solution for in-room TV advertising that balances advertiser goals with guest satisfaction.

**Key Features:**
- **Real Data:** 119,392 hotel bookings, 801 real Swiss establishments (Zurich + Lucerne)
- **Data-Driven Segmentation:** 8 guest segments from hierarchical clustering + k-means
- **Awareness Dynamics:** Segment-specific learning rates (α: 0.15-0.50, δ: 0.02-0.20)
- **Guest Experience Constraints:** 1-2 ads/day, no interruption, content filtering, federated learning
- **Reinforcement Learning:** ε-greedy RL policy training with base recommender
- **Comprehensive Evaluation:** Reach (82.4%), awareness uplift (0.287), fairness analysis, calibration

---

## Project Structure

```
.
├── src/                          # Core Python modules (33 files)
│   ├── data_loading.py          # Hotel booking data processing
│   ├── enhanced_data_loader.py   # Large dataset integration
│   ├── zurich_real_data.py      # Real Swiss advertiser data (801 establishments)
│   ├── guest_segmentation.py    # Data-driven clustering pipeline
│   ├── preferences_advanced.py  # Awareness dynamics, context modifiers
│   ├── models.py                # Baseline models (Logistic, XGBoost)
│   ├── simulation.py            # Awareness simulator
│   ├── rl_policy_training.py   # RL pipeline (5-step process)
│   ├── causal_analysis.py       # ATE, IPW, dose-response
│   ├── ablation_experiments.py  # Component ablation studies
│   ├── guest_experience_constraints.py  # Constraints implementation
│   └── ... (23 more modules)
│
├── data/
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
│
├── figures/                     # Generated figures (22 files)
│   └── qualitative/             # Awareness trajectories, dose-response, etc.
│
├── results/                     # Evaluation results (CSV files)
│
├── notebooks/                   # Jupyter notebooks for exploration
│
├── PAPER_SECTIONS.md            # Complete thesis paper (2,700+ lines)
├── latex_tables.tex             # LaTeX tables for thesis
├── latex_tables_extended.tex    # Extended analysis tables
│
├── run_segmentation_with_labels.py      # Run data-driven segmentation
├── run_complete_with_data_driven_segments.py  # Full system demo
├── run_rl_pipeline.py           # RL training pipeline
├── run_advanced_evaluation.py  # Comprehensive evaluation
├── generate_paper_figures.py   # Generate all paper figures
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data-Driven Segmentation

```bash
python run_segmentation_with_labels.py
```

This will:
- Load hotel booking data (119K bookings)
- Engineer 61 features per guest
- Run hierarchical clustering + k-means (8 segments)
- Generate segment profiles and affinities

### 3. Run Complete System

```bash
python run_complete_with_data_driven_segments.py
```

This will:
- Load 801 real Swiss establishments
- Generate guest dataset with realistic segment distribution
- Simulate 7-day stays with awareness dynamics
- Evaluate reach, awareness, scan rates

### 4. Run RL Pipeline

```bash
python run_rl_pipeline.py
```

This will:
- Train 4 baseline models (Logistic, XGBoost, Random, Popularity)
- Select best model (XGBoost) as base recommender
- Run Phase 2 simulation with full dynamics
- Train ε-greedy RL policy
- Compare base recommender vs. RL policy

### 5. Generate Paper Figures

```bash
python generate_paper_figures.py
```

Generates all figures for the thesis paper.

---

## Key Results

**Primary Metrics (N=75,166 valid bookings):**
- **Reach:** 82.4% (exceeds 80% industry target)
- **Frequency:** 4.2 exposures per guest (optimal 3-7 range)
- **Awareness Uplift:** 0.287 (exceeds 0.25 benchmark)
- **QR Scan Rate:** 1.24% (realistic for passive TV viewing)
- **Revenue/Guest:** €6.44 (+53.3% over random)

**Methodological Contributions:**
- Causal identification (endogeneity analysis, ATE estimation)
- Global sensitivity analysis (Sobol indices, interaction heatmaps)
- Cross-policy fairness analysis
- Calibration analysis (Brier score, ECE)
- Placebo validation (4 negative control tests)
- Cold-start advertiser solutions

---

## Documentation

- **`PAPER_SECTIONS.md`** - Complete thesis paper (all sections)
- **`CRITICAL_ISSUES_ACADEMIC_SUMMARY.md`** - Summary of 8 critical methodological improvements
- **`VAN_LEEUWEN_IMPROVEMENTS.md`** - Detailed improvements following van Leeuwen (2024)
- **`RL_PIPELINE_IMPLEMENTATION.md`** - RL training pipeline documentation
- **`CRITICAL_IMPROVEMENTS_APPLIED.md`** - Summary of all fixes applied

---

## Data Sources

1. **Hotel Booking Data:** 119,392 bookings from Portuguese hotels (2015-2017)
   - Source: Kaggle Hotel Booking Demand dataset
   - After filtering: 75,166 valid stays

2. **Swiss Advertiser Data:** 801 real establishments
   - Zurich Tourism: 701 establishments (6 JSON datasets)
   - Lucerne Gastronomy: 100 restaurants (from 36,085 database)
   - Categories: Experiences, Shopping, Accommodation, Restaurants, Nightlife, Wellness

3. **Weather Data:** Historical Swiss weather (MeteoSwiss API, 2024)

---

## Code Quality

- **33 Python modules** in `src/` (~5,000 lines of production code)
- **Modular architecture:** 5-layer design (Application → Business Logic → Model → Data → Utility)
- **Reproducibility:** Fixed random seeds (seed=42), versioned data
- **Documentation:** Comprehensive docstrings, type hints
- **Testing:** All modules tested and validated

---

## License

Academic research project for Master's thesis.

---

## Contact

For questions about the implementation, please refer to `PAPER_SECTIONS.md` for complete methodology documentation.
