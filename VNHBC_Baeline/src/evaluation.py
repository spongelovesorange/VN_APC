import numpy as np
import pandas as pd
from scipy.stats import rankdata, zscore

def calculate_vnhbc(judge_scores, estimated_votes, w_judge=0.55, w_vote=0.45):
    """
    Calculates the Variance-Normalized Hybrid Borda Count (VNHBC) scores.
    
    Args:
        judge_scores (np.array): Raw judge scores.
        estimated_votes (np.array): Estimated vote shares (0-1).
        w_judge (float or np.array): Weight for judges. Can be scalar or array (for Golden Paddle).
        w_vote (float): Weight for fans.
        
    Returns:
        np.array: VNHBC scores (Higher is better).
    """
    # Handle zero variance case
    if np.std(judge_scores) < 1e-9:
        z_judge = np.zeros_like(judge_scores)
    else:
        z_judge = zscore(judge_scores)
        
    if np.std(estimated_votes) < 1e-9:
        z_vote = np.zeros_like(estimated_votes)
    else:
        z_vote = zscore(estimated_votes)
        
    # Calculate weighted sum (numpy handles broadcasting if w_judge is array)
    return w_judge * z_judge + w_vote * z_vote

def calculate_consistency(judge_scores, estimated_votes, is_eliminated, rule_type):
    """
    Checks if the estimated votes + judge scores statistically explain the elimination result.
    """
    if rule_type == 'percent':
        j_share = judge_scores / (np.sum(judge_scores) + 1e-9)
        total_score = 0.5 * j_share + 0.5 * estimated_votes
    else: 
        # Rank Rule: Lower is Better. Inverted to Higher is Better (-Rank)
        j_rank = rankdata(-judge_scores, method='min')
        v_rank = rankdata(-estimated_votes, method='min')
        total_score = -1 * (j_rank + v_rank)

    elim_indices = [i for i, x in enumerate(is_eliminated) if x == 1]
    surv_indices = [i for i, x in enumerate(is_eliminated) if x == 0]
    
    if not elim_indices or not surv_indices:
        return 1.0

    # Max score of eliminated <= Min score of survivors
    max_elim_score = np.max(total_score[elim_indices])
    min_surv_score = np.min(total_score[surv_indices])
    
    if max_elim_score <= min_surv_score:
        return 1.0
    else:
        return 0.0