# Restoring the Rhythm: VN-APC Framework

## 📌 Project Overview

This repository contains the code and data for **Team #2623516**'s submission to **MCM 2026 Problem C**.

Our work addresses the "Oligarchy Paradox" in hybrid voting systems like *Dancing with the Stars*. We introduce a mathematical framework that combines **Inverse Reinforcement Learning (IRL)**, **Topological Data Analysis (TDA)**, and **Control Theory** to reconstruct latent fan preferences, diagnose systemic instability, and propose the **Variance-Normalized Adaptive Proportional Control (VN-APC)** system.

### Key Features
*   **Latent Vote Reconstruction:** Uses inverse reinforcement learning to estimate audience voting patterns from binary elimination data.
*   **Topological Audit:** Calculates persistent homology (Betti numbers) to detect "topological voids" signaling ranking anomalies.
*   **Geometric Stability Analysis:** Applies convex optimization to define the feasible space of valid scoring outcomes.
*   **VN-APC Simulation:** A PID-controlled voting mechanism that dynamically balances technical merit and popularity.

## 📂 Project Structure

```text
VN_APC/
├── main.py                 # Primary pipeline: Data loading -> IRL -> TDA -> Simulation
├── run_experiments.py      # Reproduces specific comparative experiments and figures
├── requirements.txt        # Python dependencies
├── main.tex                # LaTeX source code for the final paper
├── data/
│   ├── 2026_MCM_Problem_C_Data.csv  # Raw contest data
│   └── celebrity_fans_data.csv      # Social media/fan base metadata
├── src/                    # Core source code modules
│   ├── behavior_model.py   # Inverse Reinforcement Learning (IRL) implementation
│   ├── tda_analysis.py     # Topological Data Analysis (Betti numbers, persistence)
│   ├── data_loader.py      # Data cleaning and preprocessing
│   ├── geometric_tools.py  # Polytope construction and geometric constraints
│   ├── vn_apc_system.py    # Implementation of the VN-APC adaptive controller
│   └── utils.py            # Helper functions for visualization and logging
└── results/                # Output directory for generated tables and figures
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

*Note: Essential libraries include `numpy`, `pandas`, `scikit-learn`, `gudhi` (for TDA), `seaborn`, and `matplotlib`.*

### Running the Analysis

#### 1. Full Pipeline (`main.py`)
Run this script to go through the entire workflow described in our paper: data processing, preference learning (IRL), geometric analysis, and the primary VN-APC simulation.

```bash
python main.py
```
**Outputs:**
- `results/tables/reconstructed_votes.csv`: Estimated fan votes.
- `results/tables/season_topology.csv`: TDA metrics per season.
- `results/tables/vn_apc_simulation.csv`: Simulation results using the VN-APC mechanism.
- Various system health plots in `results/figures/`.

#### 2. Comparative Experiments (`run_experiments.py`)
Run this script to reproduce the specific comparative analysis between "Percentage", "Rank-Sum", "VN-Only", and our "VN-APC" models.

```bash
python run_experiments.py
```
**Outputs:**
- Simulation comparisons under different noise/attack levels.
- Figures demonstrating Pareto optimality and stability.

## 🧠 Methodology Highlights

1.  **Inverse Preference Learning:** We model voters as agents maximizing a "Nonlinear Underdog Utility" ($U(x) = \alpha \cdot \text{Skill} + \beta \cdot \text{Sympathy}$).
2.  **Topological Diagnostics:** We compute $\beta_1$ features from the point cloud of contestant scores to measure the "emptiness" or "conflict" in the ranking space.
3.  **Adaptive Control:** The VN-APC system uses a proportional controller ($K_p$) to adjust the weight of fan votes ($\alpha$) in real-time based on the divergence (entropy) between judge and fan rankings.

## 📄 Modeling Team

**MCM Team #2623516** (2026)
* Problem C: "Restoring the Rhythm"

---
*For questions or detailed mathematical derivations, please refer to `main.tex`.*
