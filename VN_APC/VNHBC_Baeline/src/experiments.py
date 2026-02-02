import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# === 修复：补全缺失的引用 ===
from src.inverse_solver import InverseVoter
from src.data_loader import load_and_clean_data, get_weekly_batches
from src.evaluation import calculate_consistency, calculate_vnhbc 
from src.visualize import plot_golden_paddle_impact
# ===========================

# Set global plotting style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_sensitivity_analysis(df, batches):
    """
    Exp 1: Sensitivity Analysis on Alpha.
    Fix: Changed metric from 'Consistency' to 'Vote Concentration' to show parameter impact.
    """
    print("\n[Exp 1] Running Sensitivity Analysis on Alpha (Metric: Top-1 Vote Share)...")
    
    # 扩大 alpha 范围以观察更明显的效果
    alphas = np.linspace(0.5, 3.0, 10)
    
    # 存储指标：头部选手的得票份额（衡量贫富差距）
    top_vote_shares = []
    std_shares = []

    for a in tqdm(alphas, desc="Alpha Loop"):
        # 减少 momentum_weight 以让 Zipf Prior 发挥更大作用
        solver = InverseVoter(n_samples=1000, burn_in=200, alpha_zipf=a, momentum_weight=1.0)
        
        batch_top_shares = []
        for b in batches:
            # 跳过只有2-3人的周，关注早期人多的时候
            if len(b['contestants']) < 4: continue
                
            est_votes, _ = solver.solve(b['judge_scores'], b['is_eliminated'], b['rule'])
            
            # 计算这一周最高得票者的份额
            batch_top_shares.append(np.max(est_votes))
        
        top_vote_shares.append(np.mean(batch_top_shares))
        std_shares.append(np.std(batch_top_shares))

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制主线
    ax.errorbar(alphas, top_vote_shares, yerr=std_shares, fmt='-o', capsize=5, 
                color='#348ABD', linewidth=2, ecolor='gray', label='Top-1 Vote Share')
    
    ax.set_title('Sensitivity Analysis: Impact of Zipf Parameter on Vote Concentration', fontsize=16, fontweight='bold')
    ax.set_xlabel('Zipf Exponent (Alpha)', fontsize=14)
    ax.set_ylabel('Avg. Vote Share of Top Contestant', fontsize=14)
    
    # 添加理论参考线 (如果完全平均分布，Top share ≈ 1/N，假设平均8人参赛，约为0.125)
    ax.axhline(y=0.125, color='gray', linestyle='--', label='Uniform Distribution Baseline')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp1_sensitivity_alpha_fixed.png', dpi=300)
    print(f"[Exp 1] Saved fixed plot to {OUTPUT_DIR}/exp1_sensitivity_alpha_fixed.png")
    plt.close()

def run_model_comparison(df, batches):
    """
    Exp 2: Compare Rank-Based vs. Percentage-Based Rules.
    """
    print("\n[Exp 2] Running Rule Comparison (Counterfactuals)...")
    
    solver = InverseVoter(n_samples=2000, burn_in=500, alpha_zipf=1.1, momentum_weight=15.0)
    results = []
    
    for b in tqdm(batches, desc="Batch Loop"):
        est_votes, _ = solver.solve(b['judge_scores'], b['is_eliminated'], b['rule'])
        
        rules = ['rank', 'percent']
        for r in rules:
            score = calculate_consistency(b['judge_scores'], est_votes, b['is_eliminated'], r)
            results.append({
                'season': b['season'],
                'week': b['week'],
                'rule_applied': r,
                'consistency': score,
                'actual_rule': b['rule']
            })

    res_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=res_df, x='rule_applied', y='consistency', palette='Set2', ax=ax)
    ax.set_title('Robustness Comparison: Rank vs. Percentage Rules', fontsize=16, fontweight='bold')
    ax.set_xlabel('Voting Mechanism', fontsize=14)
    ax.set_ylabel('Consistency with Historical Outcomes', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp2_rule_comparison.png', dpi=300)
    print(f"[Exp 2] Saved plot to {OUTPUT_DIR}/exp2_rule_comparison.png")
    plt.close()

def run_golden_paddle_experiment(df, batches):
    """
    Exp 3: Golden Paddle Simulation.
    Testing if doubling the judge weight for the 'victim' (high score, eliminated) saves them.
    This was MISSING in previous versions.
    """
    print("\n[Exp 3] Running Golden Paddle Simulation (Did we save the best?)...")
    
    solver = InverseVoter(n_samples=1000, burn_in=200, alpha_zipf=1.1)
    results = []
    
    for b in tqdm(batches, desc="Simulating Seasons"):
        # 1. Estimate Votes
        est_votes, _ = solver.solve(b['judge_scores'], b['is_eliminated'], b['rule'])
        
        # 2. Identify "Shock Eliminations"
        elim_indices = [i for i, x in enumerate(b['is_eliminated']) if x == 1]
        if not elim_indices: continue
            
        # Find the eliminated contestant with the highest judge score
        victim_idx = max(elim_indices, key=lambda i: b['judge_scores'][i])
        
        # Filter: Only care if they were actually good (above average judge score)
        if b['judge_scores'][victim_idx] < np.mean(b['judge_scores']):
            continue 
            
        # 3. Calculate VNHBC WITHOUT Golden Paddle (Baseline)
        scores_base = calculate_vnhbc(b['judge_scores'], est_votes, w_judge=0.55, w_vote=0.45)
        # Convert score to rank (Higher score = Rank 1)
        # Using -scores so argsort gives ascending rank (smallest is best)
        ranks_base = pd.Series(scores_base).rank(ascending=False)
        rank_base_victim = ranks_base.iloc[victim_idx]
        
        # 4. Calculate VNHBC WITH Golden Paddle
        # Logic: Head Judge doubles the weight ONLY for the victim to try and save them
        w_judge_vector = np.full(len(b['judge_scores']), 0.55)
        w_judge_vector[victim_idx] = 1.10 # DOUBLE WEIGHT
        
        scores_gp = calculate_vnhbc(b['judge_scores'], est_votes, w_judge=w_judge_vector, w_vote=0.45)
        ranks_gp = pd.Series(scores_gp).rank(ascending=False)
        rank_gp_victim = ranks_gp.iloc[victim_idx]
        
        # 5. Determine Survival
        # If rank improves to be within the number of survivors, they are "Saved"
        num_survivors = len(b['contestants']) - sum(b['is_eliminated'])
        is_saved = rank_gp_victim <= num_survivors
        
        # Only record if they were originally "out" in VNHBC too (to see the delta)
        if rank_base_victim > num_survivors:
             results.append({
                'season': b['season'],
                'week': b['week'],
                'contestant': b['contestants'][victim_idx],
                'rank_without_gp': rank_base_victim,
                'rank_with_gp': rank_gp_victim,
                'saved': is_saved
            })
            
    gp_df = pd.DataFrame(results)
    if not gp_df.empty:
        print(f"  >> Golden Paddle saved {gp_df['saved'].sum()} out of {len(gp_df)} shock eliminations.")
        gp_df.to_csv(f'{OUTPUT_DIR}/golden_paddle_results.csv', index=False)
        plot_golden_paddle_impact(gp_df)
    else:
        print("  >> No suitable shock eliminations found for Golden Paddle simulation.")

def run_ablation_study(df, batches):
    """
    Exp 4: Case Study on Bobby Bones (Season 27).
    """
    print("\n[Exp 4] Running Ablation Study (Bobby Bones Case)...")
    
    s27_batches = [b for b in batches if b['season'] == 27]
    s27_batches.sort(key=lambda x: x['week'], reverse=True)
    
    if not s27_batches: return

    target_batch = None
    bobby_idx = -1
    milo_idx = -1
    
    for batch in s27_batches:
        contestants = batch['contestants']
        b_candidates = [i for i, c in enumerate(contestants) if 'bobby' in str(c).lower()]
        m_candidates = [i for i, c in enumerate(contestants) if 'milo' in str(c).lower()]
        
        if b_candidates and m_candidates:
            target_batch = batch
            bobby_idx = b_candidates[0]
            milo_idx = m_candidates[0]
            break
    
    if target_batch is None:
        if s27_batches and len(s27_batches[0]['contestants']) >= 2:
            target_batch = s27_batches[0]
            bobby_idx, milo_idx = 0, 1
        else:
            return

    solver = InverseVoter(n_samples=2000, burn_in=500, alpha_zipf=1.1, momentum_weight=15.0)
    est_votes, _ = solver.solve(target_batch['judge_scores'], target_batch['is_eliminated'], 'percent')
    
    j_scores = target_batch['judge_scores']
    
    # Calculate Scores
    j_sum = np.sum(j_scores)
    b_j_share = j_scores[bobby_idx] / (j_sum + 1e-9)
    m_j_share = j_scores[milo_idx] / (j_sum + 1e-9)
    
    j_ranks = len(j_scores) - np.argsort(np.argsort(j_scores))
    b_j_rank = j_ranks[bobby_idx]
    m_j_rank = j_ranks[milo_idx]
    
    # VNHBC
    scores_vnhbc = calculate_vnhbc(j_scores, est_votes)
    b_vnhbc = scores_vnhbc[bobby_idx]
    m_vnhbc = scores_vnhbc[milo_idx]
    
    # Normalize for plotting
    df_plot = pd.DataFrame({
        'System': ['Percentage', 'Percentage', 'Rank', 'Rank', 'VNHBC', 'VNHBC'],
        'Contestant': [str(target_batch['contestants'][bobby_idx]), str(target_batch['contestants'][milo_idx])] * 3,
        'Raw Score': [
            0.5*b_j_share + 0.5*est_votes[bobby_idx], 
            0.5*m_j_share + 0.5*est_votes[milo_idx], 
            1.0 / (b_j_rank + 1e-9), 
            1.0 / (m_j_rank + 1e-9), 
            b_vnhbc, 
            m_vnhbc
        ]
    })
    
    df_plot['Normalized Score'] = df_plot.groupby('System')['Raw Score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9) + 0.2 
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x='System', y='Normalized Score', hue='Contestant', palette=['#E24A33', '#348ABD'], ax=ax)
    ax.set_title(f'Ablation Study: {target_batch["contestants"][bobby_idx]} vs {target_batch["contestants"][milo_idx]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/exp3_ablation.png', dpi=300)
    plt.close()