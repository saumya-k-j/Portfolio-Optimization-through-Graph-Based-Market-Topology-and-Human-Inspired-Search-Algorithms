# 📘 GL2-Robust: Graph-Based Portfolio Optimization with Human-Inspired Search

 **Research prototype for stock portfolio optimization using Graph Neural Networks (GNNs) + human-inspired search (SHLO + Hill Climbing).**  

## 1. Overview

This project implements **GL2-Robust**, a hybrid portfolio optimization framework that combines:

- **Graph Neural Networks (GNNs)** to learn structural relationships between stocks,  
- **Simple Human Learning Optimization (SHLO)** for global exploration, and  
- **Hill Climbing** for local refinement.

The GNN captures market topology (similarities across sectors, categories, and fundamentals), while SHLO/Hill-Climbing handle the search over feasible portfolios under budget and category constraints.

The system outputs:

- Optimized portfolios for four algorithm variants,  
- Objective values, budget utilization, and timing,  
- An auto-generated experiment report: **`GNN_EXPERIMENT_RESULTS.md`**.


## 2. What This Project Does

### 🔍 **Innovation**

We introduce a lightweight GNN layer that models relationships between stocks using:

1. **Category similarity**  
2. **Feature similarity** (fitness, intrinsic value gap, growth, etc.)  
3. **Message passing** (graph convolution)

The GNN produces **enhanced embeddings**, which are blended with financial features and directly influence portfolio selection.

### 🧪 **Experiment Setup**

- **Portfolio size:** 10 (can be adjusted)  
- **Category split:** e.g., 5 large-cap, 2 mid-cap, 3 small-cap  
- **Investment budget:** \$10,000  
- **Per-stock budget bounds:** 5%–20%  
- **Objective function:**

\[
J(w) = \alpha_1 \cdot \text{Health} + \alpha_2 \cdot \text{Value} + \alpha_3 \cdot \text{Growth} + \alpha_4 \cdot \text{BudgetUtilization}
\]

Default weights:

| Component                     | Weight |
|------------------------------|--------|
| Fitness Score                | 0.50   |
| Intrinsic-Value Deviation    | 0.20   |
| Revenue Growth               | 0.25   |
| Budget Utilization           | 0.05   |

Four algorithm variants are compared:

1. **Baseline SHLO**  
2. **GNN-SHLO**  
3. **Baseline Hill Climbing**  
4. **GNN-Hill Climbing**  

## 3. Method Summary

### 🧱 **Graph Construction & GNN**

**Node features include:**

- normalized fitness score  
- normalized percent difference from intrinsic value  
- normalized revenue growth  
- normalized stock price  
- category one-hot encoding  

**Edges connect stocks that:**

- belong to the same category, **or**
- have cosine similarity > 0.3

**GNN Architecture:**
Input (8 features)
→ Graph Layer (16 units, ReLU)
→ Graph Layer (8 units)
→ Embeddings


**Enhanced scoring formula:**
enhanced = 0.8 × original_score + 0.2 × embedding_score

Used for fitness, value deviation, and revenue growth.


### 🧠 SHLO (Simple Human Learning Optimization)

SHLO maintains a **population of portfolios** and updates them using:

- **Random learning** — exploration  
- **Individual learning (IKD)** — based on historical best  
- **Social learning (SKD)** — imitate global best  

Only portfolios satisfying:

- category constraints,  
- per-stock budget limits,  
- total budget ≤ \$10,000  

are accepted.


### 🧗 Hill Climbing

- Starts with a valid portfolio  
- Swaps stocks with others of the same category  
- Accepts only improvements  

The GNN-enhanced version uses the embeddings in the objective.


## 4. Repository Structure
├── data/
│ └── Final_Input_dataset_for_DSS.csv
├── notebooks/
│ ├── Intrinsic_Value_Calculation.ipynb
│ └── US_Stock_Financial_Health_Analysis.ipynb
├── src/
│ └── gnn_portfolio_experiments.py
├── GNN_EXPERIMENT_RESULTS.md
├── README.md
└── LICENSE

## 5. Installation & Setup

### 📦 Requirements

- Python **3.9+**
- Recommended: virtual environment

Install dependencies:

```bash
pip install numpy pandas jupyter

▶️ Running Experiments
cd src
python gnn_portfolio_experiments.py
This will:

load the dataset,

run SHLO/Hill-Climbing (baseline + GNN versions),

generate GNN_EXPERIMENT_RESULTS.md.

6. Using the Intrinsic-Value & Financial-Health Notebooks

Two notebooks included:

Intrinsic_Value_Calculation.ipynb

US_Stock_Financial_Health_Analysis.ipynb

These compute the financial features used by the optimizer:

intrinsic value estimates

financial-health scores

normalized growth metrics

To run them:
jupyter notebook
Then open each notebook and run all cells.

7. Configuration
All tunable settings are inside main() in the experiment script:
PORTFOLIO_SIZE = 10
LARGE_CAP_COUNT = 5
MID_CAP_COUNT = 2
TOTAL_BUDGET = 10000

UPPER_BUDGET_LIMIT = 0.20
LOWER_BUDGET_LIMIT = 0.05

WEIGHT_FITNESS = 0.5
WEIGHT_PERCENT_CHANGE = 0.2
WEIGHT_REV_GROWTH = 0.25
WEIGHT_NORMALIZED_BUDGET = 0.05

SHLO_EPOCHS = 100
SHLO_POP_SIZE = 50
HC_ITERATIONS = 1000

Random seed:

RANDOM_SEED = 42

Change these to experiment with different configurations.

8. Credits & Attribution

This project reuses and extends work from:

Suyash S. Satpute — Stock-Portfolio-Optimization-Project
🔗 https://github.com/SuyashSatpute/Stock-Portfolio-Optimization-Project

Adapted files:

Intrinsic_Value_Calculation.ipynb

US_Stock_Financial_Health_Analysis.ipynb

Licensed under MIT License, which requires proper attribution.
We extend the project with:

a new GNN architecture,

GL2-Robust optimization design,

full SHLO + GNN + Hill Climbing integration,

experimental reports.



