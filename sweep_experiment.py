import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- FILE CONFIGURATION ---
CONVKAN_FILE = "convkan_rerun_results.csv"
CNN_FILE = "overnight_sweep_results.csv"

def get_pareto_frontier(df):
    """Finds models that are optimal (best accuracy for their size category)."""
    df_sorted = df.sort_values(by='params')
    pareto_frontier = [df_sorted.iloc[0]]
    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['accuracy'] > pareto_frontier[-1]['accuracy']:
            pareto_frontier.append(df_sorted.iloc[i])
    return pd.DataFrame(pareto_frontier)

def generate_nuanced_graphics():
    if not os.path.exists(CONVKAN_FILE) or not os.path.exists(CNN_FILE):
        print("Error: Ensure both CSV files are present.")
        return

    # 1. DATA PREPARATION
    df_kan = pd.read_csv(CONVKAN_FILE).rename(columns={'width': 'width_factor'})
    df_cnn = pd.read_csv(CNN_FILE)
    
    df_kan['model_family'] = df_kan['grid_size'].apply(lambda x: f"ConvKAN (G{x})")
    df_cnn['model_family'] = "CNN"
    
    cols = ['model_family', 'depth', 'width_factor', 'params', 'accuracy', 'time_sec']
    df = pd.concat([df_kan[cols], df_cnn[cols]], ignore_index=True)

    # Calculate Efficiency Metric: Accuracy per log-parameter
    # This penalizes models that get high accuracy by just being massive.
    df['efficiency_score'] = df['accuracy'] / np.log10(df['params'])

    # Set Style
    sns.set_theme(style="white", palette="muted")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    # --- PANEL 1: THE PARETO FRONTIER (Top Left) ---
    ax1 = fig.add_subplot(gs[0, 0])
    for label in df['model_family'].unique():
        sub = df[df['model_family'] == label]
        pareto = get_pareto_frontier(sub)
        
        # Plot all points faintly
        ax1.scatter(sub['params'], sub['accuracy'], alpha=0.2, s=40)
        # Plot Pareto line boldly
        ax1.plot(pareto['params'], pareto['accuracy'], 'o-', label=label, linewidth=3, markersize=10)

    ax1.set_xscale('log')
    ax1.set_title("The Pareto Frontier: Optimal Architecture Discovery", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Parameters (Total Count)", fontsize=11)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.axvline(1000, color='red', linestyle=':', alpha=0.5)
    ax1.text(1100, 75, "1k Target", color='red', alpha=0.7)
    ax1.legend()

    # --- PANEL 2: DESIGN SPACE HEATMAP (Top Right) ---
    # We'll look at the best performing family (likely ConvKAN G5)
    ax2 = fig.add_subplot(gs[0, 1])
    kan_best = df[df['model_family'] == "ConvKAN (G5)"]
    pivot = kan_best.pivot_table(index='depth', columns='width_factor', values='accuracy')
    
    sns.heatmap(pivot, annot=True, cmap="mako", fmt=".1f", ax=ax2, cbar_kws={'label': 'Accuracy (%)'})
    ax2.set_title("ConvKAN (G5) Architecture Sensitivity: Depth vs. Width", fontsize=14, fontweight='bold')

    # --- PANEL 3: EFFICIENCY VS. COMPLEXITY (Bottom Left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=df, x="model_family", y="efficiency_score", palette="Set2", ax=ax3)
    ax3.set_title("Architecture ROI: Accuracy relative to Model Size", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Efficiency Score (Acc / log10(Params))")
    ax3.set_xlabel("")

    # --- PANEL 4: TIME-ACCURACY TRADE-OFF (Bottom Right) ---
    ax4 = fig.add_subplot(gs[1, 1])
    sns.regplot(data=df[df['model_family'] == 'CNN'], x="time_sec", y="accuracy", 
                scatter=True, label="CNN Trend", ax=ax4, lowess=True, scatter_kws={'alpha':0.3})
    sns.regplot(data=df[df['model_family'].str.contains('KAN')], x="time_sec", y="accuracy", 
                scatter=True, label="ConvKAN Trend", ax=ax4, lowess=True, scatter_kws={'alpha':0.3})
    
    ax4.set_title("Convergence Speed Comparison", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Total Training Time (s)")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend()

    plt.suptitle(f"Deep Analysis: NoduleMNIST3D Architecture Search", fontsize=20, y=0.96, fontweight='bold')
    plt.savefig('nuanced_architecture_report.png', bbox_inches='tight')
    print("Dashboard generated: nuanced_architecture_report.png")

if __name__ == "__main__":
    generate_nuanced_graphics()