import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set global style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def save_plot(fig, filename, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Saved: {path}")
    plt.close(fig)

def plot_variance_trap(var_df):
    """Plots the divergence between Judge Score Variance and Fan Vote Variance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=var_df, x='season', y='judge_score_norm', label='Judge Score Variance (Low)', 
                 color='#E24A33', linewidth=3, marker='o', ax=ax)
    sns.lineplot(data=var_df, x='estimated_vote_share', y='estimated_vote_share', label='Fan Vote Variance (High)', 
                 color='#348ABD', linewidth=3, marker='s', linestyle='--', ax=ax)
    
    ax.set_title('The "Variance Trap": Why Fan Votes Dominate in Later Seasons', fontsize=18, fontweight='bold')
    ax.set_xlabel('Season', fontsize=14)
    ax.set_ylabel('Standard Deviation (Normalized)', fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.fill_between(var_df['season'], var_df['judge_score_norm'], var_df['estimated_vote_share'], 
                    color='gray', alpha=0.1, label='Dominance Gap')
    save_plot(fig, 'fig1_variance_trap.png')

def plot_partner_effects(partner_effects, top_n=15):
    """Bar chart of the Random Effects (Pro Partner Boost)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    top_data = partner_effects.head(top_n).copy()
    sns.barplot(data=top_data, x='Effect', y='Partner', palette='viridis', ax=ax)
    ax.set_title(f'The "Kingmakers": Top {top_n} Pro Partners by Vote Boost', fontsize=16, fontweight='bold')
    ax.set_xlabel('Additional Vote Share (Standard Deviations)', fontsize=13)
    ax.set_ylabel('')
    for i, v in enumerate(top_data['Effect']):
        ax.text(v + 0.005, i + 0.1, f"+{v:.2f}", color='black', fontsize=10)
    save_plot(fig, 'fig3_partner_effects.png')

def plot_judge_save_scatter(save_df):
    """Scatter plot showing the 'Save Zone' vs 'Elimination Zone'."""
    if save_df.empty: return
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_data = []
    for idx, row in save_df.iterrows():
        plot_data.append({
            'Contestant': row['saved_contestant'],
            'Judge Score': row['saved_score'],
            'Vote Share': row['saved_vote'],
            'Outcome': 'Saved by Judges'
        })
        plot_data.append({
            'Contestant': row['sacrificed_contestant'],
            'Judge Score': row['sacrificed_score'],
            'Vote Share': row['sacrificed_vote'],
            'Outcome': 'Eliminated'
        })
    df_plot = pd.DataFrame(plot_data)
    sns.scatterplot(data=df_plot, x='Judge Score', y='Vote Share', hue='Outcome', style='Outcome',
                    palette={'Saved by Judges': '#2ecc71', 'Eliminated': '#e74c3c'}, s=100, alpha=0.8, ax=ax)
    
    ax.set_title('The "Judge\'s Save" Mechanism: Prioritizing Talent over Popularity', fontsize=16, fontweight='bold')
    ax.set_xlabel('Normalized Judge Score (Technical Merit)', fontsize=14)
    ax.set_ylabel('Estimated Fan Vote Share (Popularity)', fontsize=14)
    ax.annotate('High Skill, Low Vote\n(Target for Save)', xy=(0.8, 0.1), xytext=(0.9, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    save_plot(fig, 'fig2_judge_save_scatter.png')

def plot_case_study_trajectory(full_df, season, celebrity_name):
    """Plots the week-by-week trajectory of a specific celebrity."""
    subset = full_df[(full_df['season'] == season) & (full_df['celebrity'] == celebrity_name)].sort_values('week')
    if subset.empty: return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    weeks = subset['week']
    
    color = 'tab:red'
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Judge Score (Normalized)', color=color, fontsize=14)
    ax1.plot(weeks, subset['judge_score_norm'], color=color, marker='o', linewidth=2, label='Judge Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Est. Fan Vote Share', color=color, fontsize=14)
    ax2.plot(weeks, subset['estimated_vote_share'], color=color, marker='s', linestyle='--', linewidth=2, label='Fan Vote')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Case Study: The Divergence of {celebrity_name} (Season {season})', fontsize=16, fontweight='bold')
    if not subset.empty:
        final_vote = subset.iloc[-1]['estimated_vote_share']
        ax2.annotate(f'Final Vote: {final_vote:.1%}', xy=(weeks.iloc[-1], final_vote), xytext=(weeks.iloc[-1]-2, final_vote+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    save_plot(fig, f'fig4_case_study_{celebrity_name.replace(" ", "_")}.png')

def plot_golden_paddle_impact(gp_df):
    """
    Visualizes the impact of the Golden Paddle intervention.
    """
    if gp_df.empty: return
    
    # Take top 5 examples where Golden Paddle actually changed the outcome
    top_examples = gp_df[gp_df['saved'] == True].head(5).copy()
    if top_examples.empty:
        top_examples = gp_df.head(5).copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index = np.arange(len(top_examples))
    
    # Compare Ranks (Lower is Better)
    rects1 = ax.bar(index, top_examples['rank_without_gp'], bar_width, label='Without Golden Paddle', color='#e74c3c')
    rects2 = ax.bar(index + bar_width, top_examples['rank_with_gp'], bar_width, label='With Golden Paddle', color='#f1c40f')
    
    ax.set_xlabel('Contestant (Season-Week)')
    ax.set_ylabel('Rank (Lower is Better)')
    ax.set_title('Impact of "Golden Paddle" on Shock Eliminations', fontsize=16, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    labels = [f"{r['contestant']}\n(S{r['season']} W{r['week']})" for _, r in top_examples.iterrows()]
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    
    # Add survival line conceptual threshold (e.g. usually top 3-4 survive)
    ax.axhline(y=3.5, color='gray', linestyle='--', alpha=0.5, label='Typical Survival Threshold')
    
    save_plot(fig, 'fig5_golden_paddle_impact.png')