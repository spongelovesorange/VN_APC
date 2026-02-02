import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Import local modules
from src.data_loader import load_and_clean_data, get_weekly_batches
from src.inverse_solver import InverseVoter
from src.analysis import run_linear_mixed_model, simulate_judge_save
from src.evaluation import calculate_vnhbc
from src.visualize import plot_variance_trap, plot_partner_effects, plot_judge_save_scatter, plot_case_study_trajectory

# === 修复：确保所有实验函数都被导入 ===
from src.experiments import (
    run_golden_paddle_experiment, 
    run_sensitivity_analysis, 
    run_model_comparison, 
    run_ablation_study
)
# ==================================

# Configuration
OUTPUT_DIR = 'output'

def get_industry_prior(industry):
    """Simulates Informative Prior based on industry."""
    industry = str(industry).lower()
    if any(x in industry for x in ['reality', 'bachelor', 'kardashian']): return 0.30
    elif any(x in industry for x in ['music', 'pop', 'singer']): return 0.20
    elif any(x in industry for x in ['athlete', 'nfl', 'nba']): return 0.15
    else: return 0.10

def main():
    print(r"""
    #######################################################
       MCM 2026 PROBLEM C: DATA WITH THE STARS SOLVER
       Status: FINAL COMPLETE BUILD (NO OMISSIONS)
    #######################################################
    """)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. Data Loading
    print("[1/6] Loading Data...")
    df = load_and_clean_data('data/2026_MCM_Problem_C_Data.csv')
    
    # 2. MCMC Inference
    print("[2/6] Running Inverse MCMC Inference...")
    solver = InverseVoter(n_samples=3000, burn_in=500, alpha_zipf=1.1, momentum_weight=15.0)
    batches = get_weekly_batches(df)
    batches.sort(key=lambda x: (x['season'], x['week']))
    
    reconstructed_data = []
    history_votes = {} 

    for batch in tqdm(batches, desc="  Processing Weeks"):
        season = batch['season']
        contestants = batch['contestants']
        
        prior_means = []
        for i, name in enumerate(contestants):
            key = (season, name)
            if key in history_votes:
                prior_means.append(history_votes[key])
            else:
                prior_means.append(get_industry_prior(batch['meta'][i]['industry']))
        
        prior_means = np.array(prior_means) / (np.sum(prior_means) + 1e-9)
        
        estimated_votes, _ = solver.solve(
            batch['judge_scores'], batch['is_eliminated'], batch['rule'], prior_means
        )
        
        for i, name in enumerate(contestants):
            history_votes[(season, name)] = estimated_votes[i]
            reconstructed_data.append({
                'season': season,
                'week': batch['week'],
                'celebrity': name,
                'judge_score_norm': batch['judge_scores'][i],
                'estimated_vote_share': estimated_votes[i],
                'is_eliminated': batch['is_eliminated'][i],
                'age': batch['meta'][i]['age'],
                'industry': batch['meta'][i]['industry'],
                'partner': batch['meta'][i]['partner']
            })
            
    res_df = pd.DataFrame(reconstructed_data)
    res_df.to_csv(f'{OUTPUT_DIR}/full_simulation_results.csv', index=False)
    
    # 3. Statistical Analysis
    print("\n[3/6] Performing Statistical Analysis (LMM + Growth Rate)...")
    _, partner_effects = run_linear_mixed_model(res_df)
    if partner_effects is not None:
        partner_effects.to_csv(f'{OUTPUT_DIR}/partner_effects.csv', index=False)

    # 4. Counterfactuals: Judge Save
    print("\n[4/6] Simulating Counterfactuals: Judge Save...")
    save_df = simulate_judge_save(res_df)
    save_df.to_csv(f'{OUTPUT_DIR}/judge_save_analysis.csv', index=False)
    
    # 5. Experiments: Golden Paddle & Ablation
    print("\n[5/6] Running Advanced Experiments...")
    
    # === 修复：调用 Golden Paddle 实验 ===
    run_golden_paddle_experiment(res_df, batches)
    # ==================================
    
    run_ablation_study(res_df, batches) 
    run_sensitivity_analysis(res_df, batches)
    run_model_comparison(res_df, batches)
    
    # 6. Visualization
    print("\n[6/6] Generating Final Visualizations...")
    
    var_df = res_df.groupby(['season']).agg({
        'judge_score_norm': 'std', 'estimated_vote_share': 'std'
    }).reset_index()
    plot_variance_trap(var_df)
    plot_judge_save_scatter(save_df)
    if partner_effects is not None:
        plot_partner_effects(partner_effects)
    plot_case_study_trajectory(res_df, season=27, celebrity_name="Bobby Bones")
    plot_case_study_trajectory(res_df, season=2, celebrity_name="Jerry Rice")

    print("\n[SUCCESS] Pipeline Complete. All artifacts in '/output'. Good luck with the submission!")

if __name__ == "__main__":
    main()