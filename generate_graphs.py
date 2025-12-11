#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = 'new_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

#Results

# Current GNN experiment results
results = {
    'Baseline SHLO': {
        'objective': 5.5655,
        'budget_used': 5695,
        'budget_pct': 57.0,
        'time': 4.83
    },
    'GNN-SHLO': {
        'objective': 6.3696,
        'budget_used': 6523,
        'budget_pct': 65.2,
        'time': 4.98
    },
    'Baseline HC': {
        'objective': 7.0068,
        'budget_used': 6875,
        'budget_pct': 68.8,
        'time': 0.06
    },
    'GNN-HC': {
        'objective': 7.5926,
        'budget_used': 6351,
        'budget_pct': 63.5,
        'time': 0.09
    }
}

# Previous results from original notebooks (for comparison)
previous_results = {
    'Original SHLO': {
        'objective': 7.2907,
        'budget_used': 22983,
        'budget_pct': 76.6,
        'time': 495.50,
        'portfolio_size': 12,
        'total_budget': 30000
    },
    'Original HC': {
        'objective': 8.5614,
        'budget_used': 28728,
        'budget_pct': 95.8,
        'time': 66.65,
        'portfolio_size': 12,
        'total_budget': 30000
    }
}

# Colors
COLORS = {
    'baseline': '#3498db',      # Blue
    'gnn': '#e74c3c',           # Red
    'shlo': '#2ecc71',          # Green
    'hc': '#9b59b6',            # Purple
    'improvement': '#f39c12',   # Orange
    'original': '#95a5a6'       # Gray
}


# Graph 1: Objective Value Comparison (Bar Chart)


def plot_objective_comparison():
    """Bar chart comparing objective values across all algorithms."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algorithms = ['SHLO\n(Baseline)', 'SHLO\n(GNN-Enhanced)', 'Hill Climbing\n(Baseline)', 'Hill Climbing\n(GNN-Enhanced)']
    values = [
        results['Baseline SHLO']['objective'],
        results['GNN-SHLO']['objective'],
        results['Baseline HC']['objective'],
        results['GNN-HC']['objective']
    ]
    colors = [COLORS['baseline'], COLORS['gnn'], COLORS['baseline'], COLORS['gnn']]
    
    bars = ax.bar(algorithms, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    # Add improvement arrows
    # SHLO improvement
    ax.annotate('', xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle='->', color=COLORS['improvement'], lw=2))
    ax.text(0.5, (values[0] + values[1])/2, '+14.45%', 
            ha='center', va='center', fontsize=11, color=COLORS['improvement'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['improvement']))
    
    # HC improvement
    ax.annotate('', xy=(3, values[3]), xytext=(2, values[2]),
                arrowprops=dict(arrowstyle='->', color=COLORS['improvement'], lw=2))
    ax.text(2.5, (values[2] + values[3])/2, '+8.36%',
            ha='center', va='center', fontsize=11, color=COLORS['improvement'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['improvement']))
    
    ax.set_ylabel('Objective Function Value', fontsize=13)
    ax.set_title('GNN Enhancement: Objective Value Comparison\n(Higher is Better)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['baseline'], edgecolor='black', label='Baseline'),
                       Patch(facecolor=COLORS['gnn'], edgecolor='black', label='GNN-Enhanced')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_objective_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_objective_comparison.png")



# Graph 2: Improvement Percentage (Horizontal Bar)


def plot_improvement_percentage():
    """Horizontal bar chart showing improvement percentages."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    algorithms = ['SHLO Algorithm', 'Hill Climbing Algorithm']
    improvements = [14.45, 8.36]
    colors = [COLORS['shlo'], COLORS['hc']]
    
    bars = ax.barh(algorithms, improvements, color=colors, edgecolor='black', linewidth=1.2, height=0.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        ax.annotate(f'+{val:.2f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Improvement (%)', fontsize=13)
    ax.set_title('Performance Improvement with GNN Enhancement', fontsize=15, fontweight='bold')
    ax.set_xlim(0, max(improvements) * 1.3)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_improvement_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/2_improvement_percentage.png")


# Graph 3: Budget Utilization Comparison


def plot_budget_utilization():
    """Grouped bar chart comparing budget utilization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    baseline_budgets = [results['Baseline SHLO']['budget_pct'], results['Baseline HC']['budget_pct']]
    gnn_budgets = [results['GNN-SHLO']['budget_pct'], results['GNN-HC']['budget_pct']]
    
    bars1 = ax.bar(x - width/2, baseline_budgets, width, label='Baseline', color=COLORS['baseline'], edgecolor='black')
    bars2 = ax.bar(x + width/2, gnn_budgets, width, label='GNN-Enhanced', color=COLORS['gnn'], edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Budget Utilization (%)', fontsize=13)
    ax.set_title('Budget Utilization Comparison\n(Total Budget: $10,000)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['SHLO', 'Hill Climbing'], fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1.5, 101, 'Max Budget', ha='right', va='bottom', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_budget_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/3_budget_utilization.png")



# Graph 4: Execution Time Comparison


def plot_execution_time():
    """Bar chart comparing execution times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['Baseline\nSHLO', 'GNN-\nSHLO', 'Baseline\nHC', 'GNN-\nHC']
    times = [
        results['Baseline SHLO']['time'],
        results['GNN-SHLO']['time'],
        results['Baseline HC']['time'],
        results['GNN-HC']['time']
    ]
    colors = [COLORS['baseline'], COLORS['gnn'], COLORS['baseline'], COLORS['gnn']]
    
    bars = ax.bar(algorithms, times, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax.set_title('Execution Time Comparison\n(GNN adds minimal overhead)', fontsize=15, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['baseline'], edgecolor='black', label='Baseline'),
                       Patch(facecolor=COLORS['gnn'], edgecolor='black', label='GNN-Enhanced')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_execution_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/4_execution_time.png")



# Graph 5: Before vs After GNN (Side-by-side comparison)


def plot_before_after():
    """Side-by-side comparison showing before/after GNN enhancement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SHLO comparison
    ax1 = axes[0]
    categories = ['Objective\nValue', 'Budget\nUtilization (%)']
    before_shlo = [results['Baseline SHLO']['objective'], results['Baseline SHLO']['budget_pct']]
    after_shlo = [results['GNN-SHLO']['objective'], results['GNN-SHLO']['budget_pct']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_shlo, width, label='Before GNN', color=COLORS['baseline'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, after_shlo, width, label='After GNN', color=COLORS['gnn'], edgecolor='black')
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('SHLO Algorithm\nBefore vs After GNN', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Add improvement annotation
    ax1.annotate('+14.45%', xy=(0, after_shlo[0]), xytext=(0, after_shlo[0] + 0.5),
                fontsize=11, ha='center', color=COLORS['improvement'], fontweight='bold')
    
    # Hill Climbing comparison
    ax2 = axes[1]
    before_hc = [results['Baseline HC']['objective'], results['Baseline HC']['budget_pct']]
    after_hc = [results['GNN-HC']['objective'], results['GNN-HC']['budget_pct']]
    
    bars3 = ax2.bar(x - width/2, before_hc, width, label='Before GNN', color=COLORS['baseline'], edgecolor='black')
    bars4 = ax2.bar(x + width/2, after_hc, width, label='After GNN', color=COLORS['gnn'], edgecolor='black')
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Hill Climbing Algorithm\nBefore vs After GNN', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    
    # Add improvement annotation
    ax2.annotate('+8.36%', xy=(0, after_hc[0]), xytext=(0, after_hc[0] + 0.5),
                fontsize=11, ha='center', color=COLORS['improvement'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_before_after_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/5_before_after_comparison.png")



# Graph 6: Radar Chart - Multi-metric Comparison


def plot_radar_chart():
    """Radar chart comparing multiple metrics across algorithms."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Metrics (normalized to 0-1 scale for comparison)
    categories = ['Objective\nValue', 'Budget\nEfficiency', 'Time\nEfficiency', 'Overall\nScore']
    
    # Normalize values
    max_obj = max(results['GNN-HC']['objective'], results['GNN-SHLO']['objective'])
    
    # Calculate scores for each algorithm
    def get_scores(algo_results):
        obj_score = algo_results['objective'] / max_obj
        budget_score = algo_results['budget_pct'] / 100
        time_score = 1 - min(algo_results['time'] / 5, 1)  # Inverse, lower is better
        overall = (obj_score + budget_score + time_score) / 3
        return [obj_score, budget_score, time_score, overall]
    
    baseline_shlo_scores = get_scores(results['Baseline SHLO'])
    gnn_shlo_scores = get_scores(results['GNN-SHLO'])
    baseline_hc_scores = get_scores(results['Baseline HC'])
    gnn_hc_scores = get_scores(results['GNN-HC'])
    
    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    baseline_shlo_scores += baseline_shlo_scores[:1]
    gnn_shlo_scores += gnn_shlo_scores[:1]
    baseline_hc_scores += baseline_hc_scores[:1]
    gnn_hc_scores += gnn_hc_scores[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, baseline_shlo_scores, 'o-', linewidth=2, label='Baseline SHLO', color=COLORS['baseline'])
    ax.fill(angles, baseline_shlo_scores, alpha=0.1, color=COLORS['baseline'])
    
    ax.plot(angles, gnn_shlo_scores, 's-', linewidth=2, label='GNN-SHLO', color=COLORS['gnn'])
    ax.fill(angles, gnn_shlo_scores, alpha=0.1, color=COLORS['gnn'])
    
    ax.plot(angles, gnn_hc_scores, '^-', linewidth=2, label='GNN-HC', color=COLORS['hc'])
    ax.fill(angles, gnn_hc_scores, alpha=0.1, color=COLORS['hc'])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Multi-Metric Performance Comparison\n(Normalized Scores)', fontsize=15, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/6_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/6_radar_comparison.png")


# Graph 7: Algorithm Performance Summary


def plot_summary_dashboard():
    """Create a summary dashboard with multiple subplots."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Objective Values
    ax1 = fig.add_subplot(gs[0, 0])
    algorithms = ['Baseline\nSHLO', 'GNN-\nSHLO', 'Baseline\nHC', 'GNN-\nHC']
    objectives = [results['Baseline SHLO']['objective'], results['GNN-SHLO']['objective'],
                  results['Baseline HC']['objective'], results['GNN-HC']['objective']]
    colors = [COLORS['baseline'], COLORS['gnn'], COLORS['baseline'], COLORS['gnn']]
    ax1.bar(algorithms, objectives, color=colors, edgecolor='black')
    ax1.set_title('Objective Function Values', fontweight='bold')
    ax1.set_ylabel('Value')
    
    # Plot 2: Improvement
    ax2 = fig.add_subplot(gs[0, 1])
    improvements = [14.45, 8.36]
    ax2.bar(['SHLO', 'Hill Climbing'], improvements, color=[COLORS['shlo'], COLORS['hc']], edgecolor='black')
    ax2.set_title('GNN Improvement (%)', fontweight='bold')
    ax2.set_ylabel('Improvement (%)')
    for i, v in enumerate(improvements):
        ax2.text(i, v + 0.5, f'+{v}%', ha='center', fontweight='bold')
    
    # Plot 3: Budget Utilization
    ax3 = fig.add_subplot(gs[0, 2])
    budgets = [results['Baseline SHLO']['budget_pct'], results['GNN-SHLO']['budget_pct'],
               results['Baseline HC']['budget_pct'], results['GNN-HC']['budget_pct']]
    ax3.bar(algorithms, budgets, color=colors, edgecolor='black')
    ax3.set_title('Budget Utilization (%)', fontweight='bold')
    ax3.set_ylabel('Budget %')
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Execution Time
    ax4 = fig.add_subplot(gs[1, 0])
    times = [results['Baseline SHLO']['time'], results['GNN-SHLO']['time'],
             results['Baseline HC']['time'], results['GNN-HC']['time']]
    ax4.bar(algorithms, times, color=colors, edgecolor='black')
    ax4.set_title('Execution Time (seconds)', fontweight='bold')
    ax4.set_ylabel('Time (s)')
    
    # Plot 5: Best Algorithm Highlight
    ax5 = fig.add_subplot(gs[1, 1])
    best_data = {
        'GNN-HC\n(Best)': results['GNN-HC']['objective'],
        'GNN-SHLO': results['GNN-SHLO']['objective'],
        'Baseline HC': results['Baseline HC']['objective'],
        'Baseline SHLO': results['Baseline SHLO']['objective']
    }
    colors_best = [COLORS['improvement'], COLORS['gnn'], COLORS['baseline'], COLORS['baseline']]
    bars = ax5.barh(list(best_data.keys()), list(best_data.values()), color=colors_best, edgecolor='black')
    ax5.set_title('Algorithm Ranking', fontweight='bold')
    ax5.set_xlabel('Objective Value')
    
    # Plot 6: Key Findings Text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    findings = """
    KEY FINDINGS
    
    ✓ GNN-SHLO improved by 14.45%
    
    ✓ GNN-HC improved by 8.36%
    
    ✓ Graph-based learning
      captures stock relationships
    """
    ax6.text(0.1, 0.9, findings, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('GNN-Enhanced Portfolio Optimization: Results Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(f'{OUTPUT_DIR}/7_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/7_summary_dashboard.png")



# Graph 8: Comparison with Original Paper Results


def plot_original_comparison():
    """Compare current results with original notebook results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Note: Original results used different parameters (portfolio=12, budget=$30,000)
    # Current GNN experiments use portfolio=10, budget=$10,000
    
    # Objective Value Comparison (normalized per stock)
    ax1 = axes[0]
    
    # Normalize by portfolio size for fair comparison
    original_per_stock = [previous_results['Original SHLO']['objective'] / 12,
                          previous_results['Original HC']['objective'] / 12]
    gnn_per_stock = [results['GNN-SHLO']['objective'] / 10,
                     results['GNN-HC']['objective'] / 10]
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_per_stock, width, label='Original (12 stocks)', 
                    color=COLORS['original'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, gnn_per_stock, width, label='GNN-Enhanced (10 stocks)', 
                    color=COLORS['gnn'], edgecolor='black')
    
    ax1.set_ylabel('Objective Value per Stock', fontsize=12)
    ax1.set_title('Normalized Objective Comparison\n(Per Stock)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['SHLO', 'Hill Climbing'])
    ax1.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    # Budget Efficiency
    ax2 = axes[1]
    
    original_efficiency = [previous_results['Original SHLO']['budget_pct'],
                           previous_results['Original HC']['budget_pct']]
    gnn_efficiency = [results['GNN-SHLO']['budget_pct'],
                      results['GNN-HC']['budget_pct']]
    
    bars3 = ax2.bar(x - width/2, original_efficiency, width, label='Original', 
                    color=COLORS['original'], edgecolor='black')
    bars4 = ax2.bar(x + width/2, gnn_efficiency, width, label='GNN-Enhanced', 
                    color=COLORS['gnn'], edgecolor='black')
    
    ax2.set_ylabel('Budget Utilization (%)', fontsize=12)
    ax2.set_title('Budget Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['SHLO', 'Hill Climbing'])
    ax2.legend()
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/8_original_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/8_original_comparison.png")


def main():
    """Generate all graphs."""
    print("\n" + "="*60)
    print("   GENERATING COMPARISON GRAPHS")
    print("="*60 + "\n")
    
    plot_objective_comparison()
    plot_improvement_percentage()
    plot_budget_utilization()
    plot_execution_time()
    plot_before_after()
    plot_radar_chart()
    plot_summary_dashboard()
    plot_original_comparison()
    
    print("\n" + "="*60)
    print(f"   ALL GRAPHS SAVED TO: {OUTPUT_DIR}/")
    print("="*60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()

