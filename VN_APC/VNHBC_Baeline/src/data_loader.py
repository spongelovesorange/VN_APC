import pandas as pd
import numpy as np
import re

def parse_exit_week(result_str):
    if pd.isna(result_str): return 99
    result_str = str(result_str).lower()
    # Match "Eliminated Week X"
    match = re.search(r'eliminated.*?week\s*(\d+)', result_str)
    if match: return int(match.group(1))
    # Match finalists
    if any(x in result_str for x in ['place', 'winner', 'runner', 'finalist']): return 99
    # Handle Withdrawal/Other as non-standard elimination (treat as survival until end or ignore)
    return 99

def load_and_clean_data(filepath):
    print(f"[Info] Loading and reshaping data from {filepath}...")
    try:
        df_raw = pd.read_csv(filepath, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(filepath, encoding='utf-8')
        
    # Normalize column names
    df_raw.columns = [c.strip().lower().replace('\n', ' ').replace(' ', '_') for c in df_raw.columns]
    
    # Identify score columns
    score_cols = [c for c in df_raw.columns if 'judge' in c and 'score' in c]
    weeks = set()
    for c in score_cols:
        match = re.search(r'week(\d+)_', c)
        if match: weeks.add(int(match.group(1)))
    weeks = sorted(list(weeks))
    
    long_data = []

    for idx, row in df_raw.iterrows():
        season = row.get('season', 0)
        # Handle naming inconsistencies
        name = row.get('celebrity_name', row.get('celebrity', f"Contestant_{idx}"))
        
        # === Extract Metadata for Analysis ===
        partner = row.get('ballroom_partner', 'Unknown')
        industry = row.get('celebrity_industry', 'Unknown')
        age = row.get('celebrity_age_during_season', np.nan)
        # =====================================

        result_str = row.get('results', '')
        exit_week = parse_exit_week(result_str)
        
        for w in weeks:
            # If week is beyond exit week, skip (unless they survived)
            if w > exit_week and exit_week != 99: continue
                
            prefix = f"week{w}_"
            current_week_cols = [c for c in score_cols if prefix in c]
            
            scores = []
            for c in current_week_cols:
                val = pd.to_numeric(row[c], errors='coerce')
                if not pd.isna(val) and val > 0: scores.append(val)
            
            if not scores: continue
            
            # Normalize Scores
            total_score = sum(scores)
            # Heuristic: Max score is 40 if >30 or 4 judges, else 30
            max_possible = 40 if (len(scores) > 3 or total_score > 30) else 30
            norm_score = total_score / max_possible
            
            is_eliminated = (w == exit_week)
            # Define Rules based on Season
            rule_type = 'rank' if (season <= 2 or season >= 28) else 'percent'
            
            long_data.append({
                'season': season,
                'week': w,
                'celebrity': name,
                'partner': partner,
                'industry': industry,
                'age': age,
                'judge_score_norm': norm_score,
                'is_eliminated': 1 if is_eliminated else 0,
                'rule_type': rule_type,
                'original_result': result_str
            })
            
    df_long = pd.DataFrame(long_data)
    print(f"[Info] Reshaping complete. Converted to {len(df_long)} weekly records.")
    return df_long

def get_weekly_batches(df):
    batches = []
    df = df.sort_values(['season', 'week'])
    for (season, week), group in df.groupby(['season', 'week']):
        if len(group) < 2: continue
        eliminated_count = group['is_eliminated'].sum()
        if eliminated_count == 0: continue
            
        batches.append({
            'season': season,
            'week': week,
            'rule': group['rule_type'].iloc[0],
            'contestants': group['celebrity'].tolist(),
            'judge_scores': group['judge_score_norm'].values,
            'is_eliminated': group['is_eliminated'].values,
            'meta': group[['age', 'industry', 'partner']].to_dict('records')
        })
    return batches