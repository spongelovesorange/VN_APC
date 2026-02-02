import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_and_clean_data
from src.behavior_model import InversePreferenceLearner
from src.tda_analysis import analyze_season_topology
from src.geometric_tools import analyze_geometric_constraints
from src.vn_apc_system import run_vn_apc_simulation
from src.utils import setup_env, save_figure

def main():
    setup_env()
    print("=== MCM 2026 Problem C: O-Prize Solution Pipeline ===")
    
    # 1. Load Data
    fan_data_path = 'data/celebrity_fans_data.csv'
    # Fallback to absolute path if needed
    if not os.path.exists(fan_data_path):
        fan_data_path = '/Users/beaulocanana/Desktop/2026_MCM_Integrated/data/celebrity_fans_data.csv'

    df = load_and_clean_data(
            'data/2026_MCM_Problem_C_Data.csv', 
            fan_filepath=fan_data_path
        )
    
    # 2. Preference Learning (Inverse RL)
    learner = InversePreferenceLearner()
    df = learner.train_and_predict_walk_forward(df)
    
    # Save intermediate results (Behavioral Model)
    df.to_csv('results/tables/reconstructed_votes.csv', index=False)
    
    # Visualize Weights
    plt.figure(figsize=(10, 6))
    sns.barplot(x=learner.feature_names, y=learner.weights)
    plt.title("Learned Audience Weights")
    plt.xticks(rotation=45)
    save_figure('fig1_weights.png')
    
    # 3. Geometric Analysis (Linear Programming / Chebyshev Center)
    # Pass learner to enable weighted network constraints in LP
    print("[System] Running Geometric Constraints Analysis...")
    geo_res = analyze_geometric_constraints(df, learner=learner)
    
    # Save geometric metrics and updated df with robust votes
    geo_res.to_csv('results/tables/geometric_feasibility.csv', index=False)
    df.to_csv('results/tables/reconstructed_votes_robust.csv', index=False)
    
    # 4. Topological Analysis
    topo_res = analyze_season_topology(df)
    topo_res.to_csv('results/tables/season_topology.csv', index=False)
    
    # Visualize Topology Health
    plt.figure(figsize=(12, 6))
    if not topo_res.empty:
        sns.regplot(data=topo_res, x='Season', y='Topo_Health', order=2)
        plt.title("Evolution of System Topological Health")
        save_figure('fig2_topo_health.png')
    
    # 5. VN-APC Simulation
    df_sim = run_vn_apc_simulation(df, topo_res, geo_res)
    df_sim.to_csv('results/tables/vn_apc_simulation.csv', index=False)
    
    # 6. Adaptive Control Visualization
    if not topo_res.empty:
        worst_s = topo_res.sort_values('Topo_Health').iloc[0]['Season']
        subset = df_sim[df_sim['Season'] == worst_s]
        week_alpha = subset.groupby('Week')['Adaptive_Alpha'].mean().reset_index()
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=week_alpha, x='Week', y='Adaptive_Alpha', marker='o')
        plt.title(f"Adaptive Control Response (Season {worst_s})")
        plt.ylabel("Judge Weight (Alpha)")
        save_figure('fig3_adaptive_control.png')
        
    print("[Success] All pipeline stages completed.")

if __name__ == "__main__":
    main()