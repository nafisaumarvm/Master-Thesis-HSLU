# In-Room TV Advertising Recommender System

**Nafisa Umar Master's Thesis HSLU**
---

## Overview

This system implements a complete pipeline for in-room TV advertising in hotels that optimizes for visibility, reach and awareness building.

**Key Features:**
- **Real Data:** 119,392 hotel bookings, 616 real Swiss establishments (Zurich + Lucerne)
- **Data-Driven Segmentation:** 8 guest segments from hierarchical clustering + k-means
- **Awareness Dynamics:** Segment-specific learning rates (α: 0.15-0.50, δ: 0.02-0.20)
- **Guest Experience Constraints:** 1-2 ads/day, no interruption, content filtering
  
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

### 3. Run Complete System

```bash
python run_complete_with_data_driven_segments.py
```

### 4. Run Full Evaluation

```bash
python run_full_evaluation_fixed.py
```

---

## Key Results

**Primary Metrics (N=75,166 valid bookings, 51,113 TV-watching guests):**
- **Reach:** 83.54%
- **Frequency:** 4.10 exposures per guest
- **Awareness Uplift:** 0.230
- **QR Scan Rate:** 1.07% (realistic for passive TV viewing)
- **GRP:** 342.1 

---

## Data Sources

1. **Hotel Booking Data:** 119,392 bookings from Portuguese hotels (2015-2017)
   - After filtering: 75,166 valid stays
   - TV-watching guests: 51,113 (68% viewing rate)

2. **Swiss Advertiser Data:** 616 real establishments
   - Zurich Tourism: 516 establishments (5 JSON datasets)
   - Lucerne Gastronomy: 100 restaurants (from 36,085 database)
   - Categories: Experiences (240), Shopping (174), Restaurants (119), Nightlife (55), Wellness (28)

3. **Weather Data:** Historical Swiss weather (MeteoSwiss API, 2024)

---

## Project Structure

```
.
├── src/                          # Core Python modules
│   ├── data_loading.py          # Hotel booking data processing
│   ├── zurich_real_data.py      # Real Swiss advertiser data (616 establishments)
│   ├── guest_segmentation.py    # Data-driven clustering pipeline
│   ├── preferences_advanced.py  # Awareness dynamics
│   ├── models.py                # Baseline models
│   ├── simulation.py            # Awareness simulator
│   └── ... (28 more modules)
│
├── data/                        # Raw and processed datasets
├── results/                     # Evaluation results
├── notebooks/                   # Jupyter notebooks
│
├── run_full_evaluation_fixed.py # Main evaluation script
└── requirements.txt             # Python dependencies
```

---

