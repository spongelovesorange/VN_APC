import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def clean_partner_name(name):
    """
    Cleans partner names. 
    Example: 'Emma Slater/Kaitlyn Bristowe' -> 'Emma Slater'
    """
    if pd.isna(name):
        return "Unknown"
    return str(name).split('/')[0].strip()

def run_linear_mixed_model(df):
    """
    Optimized LMM with Feature Engineering (Growth Rate) and Robust Optimization.
    """
    print("[ANALYSIS] Initializing Linear Mixed Model (LMM)...")
    
    data = df.copy()
    
    # 1. Data Cleaning
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    data['partner'] = data['partner'].apply(clean_partner_name) 
    
    # 2. Feature Engineering: Growth Rate
    # Sort to ensure correct shift (Previous week's score)
    data = data.sort_values(['season', 'celebrity', 'week'])
    data['prev_score'] = data.groupby(['season', 'celebrity'])['judge_score_norm'].shift(1)
    data['growth_rate'] = data['judge_score_norm'] - data['prev_score']
    # Fill NA for first week (growth is 0)
    data['growth_rate'] = data['growth_rate'].fillna(0) 
    
    data = data.dropna(subset=['age', 'industry', 'partner', 'estimated_vote_share'])
    
    # 3. Filter Rare Partners (Stabilize Convergence)
    partner_counts = data['partner'].value_counts()
    valid_partners = partner_counts[partner_counts >= 5].index 
    data = data[data['partner'].isin(valid_partners)]
    
    # 4. Standardize Target
    data['vote_z'] = data.groupby(['season', 'week'])['estimated_vote_share'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    
    # 5. Industry Encoding
    top_industries = data['industry'].value_counts().nlargest(5).index
    data['industry_enc'] = data['industry'].apply(lambda x: x if x in top_industries else 'Other')
    
    # 6. Model Specification (Added growth_rate)
    # We test if 'improving' (positive growth) correlates with higher votes
    formula = "vote_z ~ age + C(industry_enc) + judge_score_norm + growth_rate"
    
    try:
        print(f"[ANALYSIS] Fitting model on {len(data)} observations with {len(valid_partners)} unique partners...")
        
        # Using LBFGS optimizer for robust convergence
        model = smf.mixedlm(formula, data, groups=data['partner'])
        result = model.fit(method='lbfgs', maxiter=2000)
        
        print("\n" + "="*50)
        print("          LMM REGRESSION RESULTS (OPTIMIZED)      ")
        print("="*50)
        print(result.summary())
        
        re_dict = {k: v['Group'] for k, v in result.random_effects.items()} 
        partner_effects = pd.DataFrame(list(re_dict.items()), columns=['Partner', 'Effect'])
        partner_effects = partner_effects.sort_values('Effect', ascending=False)
        
        return result, partner_effects
        
    except Exception as e:
        print(f"[ERROR] LMM Fitting Failed: {e}")
        return None, None

def simulate_judge_save(df):
    """
    Simulates Judges' Save.
    Logic: The judge saves the contestant with the higher technical score.
    This aligns with the 'Head Judge' weighting preference for technique.
    """
    print("[ANALYSIS] Simulating Counterfactual: 'The Judges' Save'...")
    
    results = []
    
    df = df.sort_values(['season', 'week'])

    for (season, week), group in df.groupby(['season', 'week']):
        if group['is_eliminated'].sum() == 0: continue
        
        eliminated_actual = group[group['is_eliminated'] == 1]
        
        # Calculate Percentage Rule (S3-S27 baseline for most controversies)
        j_share = group['judge_score_norm'] / (group['judge_score_norm'].sum() + 1e-9)
        v_share = group['estimated_vote_share']
        total_score = 0.5 * j_share + 0.5 * v_share
        
        # Identify Bottom 2
        bottom_2 = total_score.nsmallest(2)
        bottom_2_indices = bottom_2.index
        
        if len(bottom_2_indices) < 2: continue 

        for idx, row in eliminated_actual.iterrows():
            if idx in bottom_2_indices:
                # Find opponent in Bottom 2
                opponent_idx = bottom_2_indices[0] if bottom_2_indices[1] == idx else bottom_2_indices[1]
                opponent_row = group.loc[opponent_idx]
                
                # DECISION RULE: Judges save the one with higher judge score
                if row['judge_score_norm'] > opponent_row['judge_score_norm']:
                    results.append({
                        'season': season,
                        'week': week,
                        'saved_contestant': row['celebrity'],
                        'saved_score': row['judge_score_norm'],
                        'saved_vote': row['estimated_vote_share'],
                        'sacrificed_contestant': opponent_row['celebrity'],
                        'sacrificed_score': opponent_row['judge_score_norm'],
                        'sacrificed_vote': opponent_row['estimated_vote_share']
                    })
                    
    print(f"[RESULT] The Judges' Save would have altered {len(results)} historical outcomes.")
    return pd.DataFrame(results)