import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore, rankdata

# ==========================================
# 1. Environment & Utils
# ==========================================

def setup_env():
    # 优化配色方案：增加紫色代表"Rank System"
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    # Gray(Legacy/Percent), Purple(Rank), Blue(VN-Only), Red(VN-APC/Ours)
    custom_palette = ["#999999", "#988ED5", "#348ABD", "#E24A33"] 
    sns.set_palette(custom_palette)
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    print("[Exp] Environment setup complete.")

def save_figure(filename, dpi=300):
    path = os.path.join('results/figures', filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[Exp] Saved figure: {path}")

def safe_rank_corr(x, y):
    if len(x) < 2: return 0.0
    if pd.Series(x).nunique() <= 1 or pd.Series(y).nunique() <= 1: return 0.0
    return pd.Series(x).rank().corr(pd.Series(y).rank())

# ==========================================
# 2. Simulation Engine (Enhanced)
# ==========================================

def simulate_variant(df_input, mode='Full', kp=0.6, ki=0.05, noise_level=0.0, override_vote_col=None):
    """
    核心模拟引擎，支持 Percent, Rank, VN_Only, Full (VN-APC) 四种模式
    """
    df = df_input.copy()
    df['Sim_Score'] = 0.0
    df = df.sort_values(['Season', 'Week'])
    
    current_alpha = 0.5
    integral_error = 0.0
    
    vote_col = override_vote_col if override_vote_col else 'Est_Fan_Vote'
    
    for (s, w), group in df.groupby(['Season', 'Week'], sort=False):
        tech = group['Tech_Score'].values
        vote = group[vote_col].values.copy() 
        n_contestants = len(tech)
        
        # --- A. Attack Injection (Adversarial) ---
        if noise_level > 0 and len(vote) > 0:
            # 攻击者尝试给技术分最低的人刷票
            worst_tech_idx = np.argmin(tech)
            noise = np.random.normal(0, noise_level * 0.2, size=len(vote)) 
            noise[worst_tech_idx] += noise_level 
            vote = np.clip(vote + noise, 0, None)
            # 重新归一化
            if np.sum(vote) > 0: vote /= np.sum(vote)
        
        # --- B. Algorithm Logic ---
        score = np.zeros_like(tech)
        
        if mode == 'Percent':
            # 百分比制：直接加权 (假设 Tech 满分40，Vote 比例放大)
            # 为了公平对比，我们先归一化 Tech (share) 和 Vote (share)
            tech_share = tech / (np.sum(tech) + 1e-9)
            vote_share = vote / (np.sum(vote) + 1e-9)
            score = 0.5 * tech_share + 0.5 * vote_share
            
        elif mode == 'Rank':
            # === [Q2 Key] 排名制逻辑 ===
            # Rank 1 is Best. rankdata default: small number = small value.
            # So we rank (-score), so highest score gets rank 1.
            rank_tech = rankdata(-tech, method='min')
            rank_vote = rankdata(-vote, method='min')
            
            sum_ranks = rank_tech + rank_vote
            # 注意：Sim_Score 越高越好，但 Rank Sum 越小越好。
            # 取负数以便后续统一处理 (Correlation, Sorting)
            score = -1.0 * sum_ranks 
            
        elif mode == 'VN_Only':
            # 纯 Z-Score，无动态调整
            z_tech = zscore(tech) if np.std(tech) > 1e-9 else tech
            z_vote = zscore(vote) if np.std(vote) > 1e-9 else vote
            z_tech = np.nan_to_num(z_tech)
            z_vote = np.nan_to_num(z_vote)
            score = 0.5 * z_tech + 0.5 * z_vote
            
        elif mode == 'Full':
            # VN-APC (PID Control)
            if n_contestants > 2 and np.std(tech) > 0:
                corr = np.corrcoef(tech, vote)[0, 1]
                if np.isnan(corr): corr = 0
            else:
                corr = 0.8
            
            # [Fix: High Target] 提高目标到 0.96，防止 PID 反向优化
            target_corr = 0.96
            error = target_corr - corr
            integral_error = integral_error * 0.9 + error
            
            delta = kp * error + ki * integral_error
            delta += np.random.normal(0, 0.005) # Dither
            delta = np.clip(delta, -0.2, 0.2)
            
            current_alpha = np.clip(current_alpha + delta, 0.2, 0.95)
            
            z_tech = zscore(tech) if np.std(tech) > 1e-9 else tech
            z_vote = zscore(vote) if np.std(vote) > 1e-9 else vote
            z_tech = np.nan_to_num(z_tech)
            z_vote = np.nan_to_num(z_vote)
            
            score = current_alpha * z_tech + (1 - current_alpha) * z_vote
            
        df.loc[group.index, 'Sim_Score'] = score
        
    return df

# ==========================================
# 3. Experiments
# ==========================================

def run_q2_mechanism_audit(df):
    """
    [Question 2 Answer]
    对比 Rank vs Percent 在特定争议人物（Bobby Bones, Jerry Rice）上的表现。
    生成详细 CSV 报告。
    """
    print("[Exp] Running Q2 Mechanism Audit (Rank vs Percent)...")
    
    # 1. Run both simulations
    res_percent = simulate_variant(df, mode='Percent')
    res_rank = simulate_variant(df, mode='Rank')
    
    # 2. Define Controversial Cases
    targets = ['Bobby Bones', 'Jerry Rice', 'Bristol Palin', 'Billy Ray Cyrus']
    
    audit_data = []
    
    for name in targets:
        # 模糊匹配名字
        mask = df['Contestant'].str.contains(name, case=False, na=False)
        target_seasons = df.loc[mask, 'Season'].unique()
        
        for s in target_seasons:
            weeks = df[df['Season'] == s]['Week'].unique()
            
            for w in sorted(weeks):
                # 获取该周数据
                p_week = res_percent[(res_percent['Season'] == s) & (res_percent['Week'] == w)]
                r_week = res_rank[(res_rank['Season'] == s) & (res_rank['Week'] == w)]
                
                if p_week.empty: continue
                
                # 找到目标人物
                p_row = p_week[p_week['Contestant'].str.contains(name, case=False, na=False)]
                if p_row.empty: 
                    # 可能已经被淘汰了
                    status = "Eliminated"
                    rank_p = -1
                    rank_r = -1
                else:
                    status = "Active"
                    # 计算在 Percent 系统中的排名 (Sim_Score 越高越好 -> Rank 1)
                    # rankdata gives small number for small value, so rank(-score)
                    p_ranks = rankdata(-p_week['Sim_Score'].values, method='min')
                    p_idx = np.where(p_week.index == p_row.index[0])[0][0]
                    rank_p = p_ranks[p_idx]
                    
                    # 计算在 Rank 系统中的排名
                    r_row = r_week[r_week['Contestant'].str.contains(name, case=False, na=False)]
                    r_ranks = rankdata(-r_week['Sim_Score'].values, method='min')
                    r_idx = np.where(r_week.index == r_row.index[0])[0][0]
                    rank_r = r_ranks[r_idx]
                    
                    # 记录
                    audit_data.append({
                        'Contestant': name,
                        'Season': s,
                        'Week': w,
                        'Tech_Score': p_row['Tech_Score'].values[0],
                        'Est_Fan_Vote': p_row['Est_Fan_Vote'].values[0],
                        'Rank_in_Percent_System': rank_p,
                        'Rank_in_Rank_System': rank_r,
                        'Diff': rank_r - rank_p # 正值意味着 Rank 系统排名更靠后（更差），负值意味着更好
                    })
    
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv('results/tables/q2_mechanism_audit.csv', index=False)
    print("[Exp] Saved Q2 Audit Table: results/tables/q2_mechanism_audit.csv")

def run_judges_choice_impact(df):
    """
    [Question 2 Answer]
    分析 "Judge's Choice" (Bottom 2 Saving) 机制的影响。
    如果 Bottom 2 中，得分较低的人其实 Tech Score 更高，则视为"裁判拯救成功"。
    """
    print("[Exp] Analyzing Judge's Choice (Bottom 2) Impact...")
    
    # 我们基于 Percent 系统来模拟（因为 S28 之前是 Percent）
    res = simulate_variant(df, mode='Percent')
    
    saved_count = 0
    total_eliminations = 0
    
    for (s, w), group in res.groupby(['Season', 'Week']):
        if len(group) < 3: continue # 决赛周通常不适用
        
        # 1. 找出 Percent 系统下的 Bottom 2
        # Sim_Score 越低越差
        sorted_grp = group.sort_values('Sim_Score', ascending=True)
        bottom_2 = sorted_grp.iloc[:2]
        
        loser = bottom_2.iloc[0] # 本该被淘汰的人 (最低分)
        runner_up = bottom_2.iloc[1] # 倒数第二
        
        total_eliminations += 1
        
        # 2. 裁判逻辑：谁的 Tech Score 高，谁留下
        # 如果 loser 的技术分 明显高于 (> 0.05) runner_up，裁判会救 loser
        if loser['Tech_Score'] > runner_up['Tech_Score'] + 0.05:
            saved_count += 1
            
    print(f"[Result] Judge's Choice Analysis:")
    print(f"  - Total Elimination Rounds: {total_eliminations}")
    print(f"  - Potential Judge Saves (Reversed Outcome): {saved_count}")
    print(f"  - Impact Rate: {saved_count/total_eliminations*100:.2f}%")
    
    # 保存简单统计
    with open('results/tables/q2_judges_choice_stats.txt', 'w') as f:
        f.write(f"Total Rounds: {total_eliminations}\n")
        f.write(f"Reversed Outcomes: {saved_count}\n")
        f.write(f"Impact Rate: {saved_count/total_eliminations:.4f}\n")

def run_ground_truth_verification(df):
    """
    [Core Verification]
    平行宇宙实验：加入 Rank System 对比
    """
    print("[Exp] Running Parallel Universe Verification (Rank vs Percent vs VN-APC)...")
    
    scenarios = ['GT_Merit', 'GT_Chaos', 'GT_Oligarchy']
    results = []
    
    df_univ = df.copy()
    np.random.seed(2026)
    n = len(df_univ)
    
    # Construct Universes
    df_univ['GT_Merit'] = df_univ['Tech_Score'] + np.random.normal(0, 0.05, n)
    df_univ['GT_Chaos'] = np.random.rand(n)
    
    oligarchs = np.random.choice(df_univ['Contestant'].unique(), 3, replace=False)
    df_univ['GT_Oligarchy'] = np.random.rand(n) * 0.2
    mask = df_univ['Contestant'].isin(oligarchs)
    df_univ.loc[mask, 'GT_Oligarchy'] += 0.8 # 寡头获得巨额票数
    
    modes_to_test = ['Percent', 'Rank', 'Full']
    mode_labels = {'Percent': 'Legacy (%)', 'Rank': 'Rank Sys', 'Full': 'VN-APC'}
    
    for sc in scenarios:
        for m in modes_to_test:
            res = simulate_variant(df_univ, mode=m, override_vote_col=sc)
            
            # Metric: Correlation between Tech and Final Outcome
            metric = np.mean([safe_rank_corr(g['Tech_Score'], g['Sim_Score']) 
                              for _, g in res.groupby(['Season', 'Week'])])
            
            results.append({
                'Scenario': sc, 
                'System': mode_labels[m], 
                'Fairness': metric
            })
    
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    labels = {'GT_Merit': 'Meritocracy', 'GT_Chaos': 'Chaos', 'GT_Oligarchy': 'Oligarchy (Biased)'}
    res_df['Scenario_Label'] = res_df['Scenario'].map(labels)
    
    sns.barplot(data=res_df, x='Scenario_Label', y='Fairness', hue='System')
    
    plt.title("System Robustness: Rank vs Percent vs VN-APC", fontsize=16)
    plt.ylabel("Meritocratic Consistency (Tech-Result Corr)", fontsize=13)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper right', title='Voting Mechanism')
    
    # Labeling
    for p in plt.gca().patches:
        h = p.get_height()
        text = f'{h:.2f}' if h > 0.01 else '0.00'
        plt.gca().annotate(text, (p.get_x() + p.get_width() / 2., max(h, 0)), 
                           ha='center', va='bottom', xytext=(0, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
                           
    save_figure('exp4_parallel_universes.png')

def run_ablation_study(df):
    """
    消融实验：加入 Rank System
    """
    print("[Exp] Running Ablation Study (Legacy vs Rank vs VN-APC)...")
    modes = ['Percent', 'Rank', 'VN_Only', 'Full']
    results = {}
    
    for m in modes:
        res = simulate_variant(df, mode=m)
        corrs = [safe_rank_corr(g['Tech_Score'], g['Sim_Score']) for _, g in res.groupby(['Season', 'Week'])]
        results[m] = np.mean(corrs)
    
    plt.figure(figsize=(10, 6))
    # Colors: Gray, Purple, Blue, Red
    colors = ['#BBBBBB', '#988ED5', '#348ABD', '#E24A33']
    bars = plt.bar(results.keys(), results.values(), color=colors, width=0.6)
    
    plt.title("Ablation Study: Mechanism Comparison", fontsize=15)
    plt.ylabel("Average Fairness (Spearman Corr)", fontsize=12)
    plt.xticks(range(4), ["Legacy (%)", "Rank Sys", "VN Only", "VN-APC (Full)"])
    plt.ylim(0, 1.1)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha='center', fontsize=12, fontweight='bold')
    save_figure('exp1_ablation_study.png')

def run_vector_field_analysis(df):
    """
    VN-APC 控制动力学相图
    """
    print("[Exp] Generating Vector Field (Control Dynamics)...")
    x = np.linspace(0, 1.0, 20) 
    y = np.linspace(0.1, 0.9, 20) 
    X, Y = np.meshgrid(x, y)
    
    Target_Disorder = 0.05 
    Kp = 0.8
    
    # d(Alpha)/dt = Kp * error
    # error = Target_Corr - Current_Corr
    # Disorder = 1 - Current_Corr
    # So error = Target_Corr - (1 - Disorder)
    # Let's simplify visual: Higher Disorder -> Higher Alpha needed (but careful)
    # Actually logic: if Correlation is Low (Disorder High), we increase Alpha (Judge Weight).
    
    V = Kp * (X - Target_Disorder) # d(Alpha)
    U = -1.5 * (Y - 0.5) # d(Disorder) - assumed natural decay towards medium
    
    plt.figure(figsize=(10, 7))
    plt.quiver(X, Y, U, V, color='#348ABD', alpha=0.8, scale=20)
    plt.axvline(x=Target_Disorder, color='red', linestyle='--', linewidth=2, label='Target Equilibrium')
    
    plt.title("System Phase Space: Control Vector Field", fontsize=16)
    plt.xlabel("System Disorder (1 - Correlation)", fontsize=14)
    plt.ylabel("Control Output (Judge Weight $\\alpha$)", fontsize=14)
    plt.legend(loc='upper right')
    save_figure('exp0_vector_field.png')

def run_phase_portrait_analysis(df):
    """
    Empirical Phase Portrait (Before vs After)
    """
    print("[Exp] Generating Empirical Phase Portrait...")
    
    temp_res = simulate_variant(df, mode='Full')
    df['VN_APC_Score'] = temp_res['Sim_Score']
    sample = df.sample(min(len(df), 800), random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Before Control
    sns.scatterplot(ax=axes[0], data=sample, x='Tech_Score', y='Est_Fan_Vote', 
                    alpha=0.5, color='#999999', s=60, linewidth=0)
    axes[0].set_title("Before Control: Entropy", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Technical Merit")
    axes[0].set_ylabel("Fan Vote")
    
    # After Control
    y_vals = sample['VN_APC_Score']
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
    
    sns.scatterplot(ax=axes[1], x=sample['Tech_Score'], y=y_norm, 
                    alpha=0.6, color='#E24A33', s=60, linewidth=0)
    axes[1].set_title("After VN-APC: Alignment", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Technical Merit")
    axes[1].set_ylabel("Final Score (Norm)")
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    
    save_figure('exp0_phase_portrait_scatter.png')

def run_sensitivity_heatmap(df):
    print("[Exp] Running Sensitivity Analysis...")
    k_p_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    means, stds = [], []

    for kp in k_p_vals:
        res = simulate_variant(df, mode='Full', kp=kp)
        corrs = [safe_rank_corr(g['Tech_Score'], g['Sim_Score']) for _, g in res.groupby(['Season', 'Week'])]
        means.append(np.mean(corrs))
        stds.append(np.std(corrs))
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_p_vals, means, 'o-', color='#348ABD', linewidth=3, markersize=8)
    plt.fill_between(k_p_vals, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color='#348ABD', alpha=0.2)
    plt.title("Parameter Sensitivity (Kp)", fontsize=14)
    plt.xlabel("Kp Value")
    plt.ylabel("System Fairness")
    save_figure('exp2_sensitivity_kp.png')

def run_adversarial_robustness(df):
    print("[Exp] Running Adversarial Robustness Check...")
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    perf_base, perf_rank, perf_ours = [], [], []
    
    for n in noise_levels:
        # Legacy
        res_base = simulate_variant(df, mode='Percent', noise_level=n)
        perf_base.append(np.mean([safe_rank_corr(g['Tech_Score'], g['Sim_Score']) for _, g in res_base.groupby(['Season', 'Week'])]))
        
        # Rank System (Also check this!)
        res_rank = simulate_variant(df, mode='Rank', noise_level=n)
        perf_rank.append(np.mean([safe_rank_corr(g['Tech_Score'], g['Sim_Score']) for _, g in res_rank.groupby(['Season', 'Week'])]))

        # Ours
        res_ours = simulate_variant(df, mode='Full', noise_level=n)
        perf_ours.append(np.mean([safe_rank_corr(g['Tech_Score'], g['Sim_Score']) for _, g in res_ours.groupby(['Season', 'Week'])]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, perf_base, 'o--', color='gray', label='Legacy (%)', linewidth=2)
    plt.plot(noise_levels, perf_rank, 's-.', color='#988ED5', label='Rank Sys', linewidth=2)
    plt.plot(noise_levels, perf_ours, 'D-', color='#E24A33', label='VN-APC (Ours)', linewidth=3)
    
    plt.title("System Robustness Under Voting Attacks", fontsize=16)
    plt.xlabel("Attack Intensity (Sigma)")
    plt.ylabel("Fairness Retention")
    plt.legend()
    save_figure('exp3_robustness.png')

# ==========================================
# 4. Main
# ==========================================

def main():
    setup_env()
    print("=== Core-Grade Experiments: O-Prize Verification ===")
    
    data_path = 'results/tables/reconstructed_votes.csv'
    if not os.path.exists(data_path):
        print(f"[Error] {data_path} not found. Run main.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Core Physics
    run_vector_field_analysis(df)
    run_phase_portrait_analysis(df)
    
    # 2. Key Verifications (Q2 & Q4)
    run_q2_mechanism_audit(df)       # NEW: Bobby Bones / Jerry Rice specifics
    run_judges_choice_impact(df)     # NEW: Judge's Save Analysis
    run_ground_truth_verification(df) # Updated with Rank
    
    # 3. System Analysis
    run_ablation_study(df)            # Updated with Rank
    run_sensitivity_heatmap(df)
    run_adversarial_robustness(df)
    
    print("[Success] All experiments completed.")

if __name__ == "__main__":
    main()