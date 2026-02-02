import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform

class PolytopeSolver:
    """
    Solves for the Chebyshev Center of the feasible voting polytope defined by:
    1. Simplex constraints (sum = 1)
    2. Elimination constraints (survivor > eliminated)
    3. Network structure constraints (vote diff <= feature dist)
    """
    def find_chebyshev_center(self, A_ub, b_ub, A_eq, b_eq, n_vars):
        # Objective: Maximize radius r -> Minimize -r
        # Variables: x = [v_1, ..., v_n, r] (n+1 dimensions)
        c = np.zeros(n_vars + 1)
        c[-1] = -1.0 

        # Transform inequalities: A_i * x + ||A_i|| * r <= b_i
        row_norms = np.linalg.norm(A_ub, axis=1).reshape(-1, 1)
        A_ub_cheby = np.hstack([A_ub, row_norms])
        
        # Transform equalities: A_eq * x + 0 * r = b_eq
        if A_eq is not None:
            A_eq_cheby = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        else:
            A_eq_cheby = None

        # Bounds: v_i in [0, 1], r >= 0
        bounds = [(0, 1) for _ in range(n_vars)] + [(0, None)]

        try:
            res = linprog(
                c, 
                A_ub=A_ub_cheby, b_ub=b_ub,
                A_eq=A_eq_cheby, b_eq=b_eq,
                bounds=bounds,
                method='highs' # Robust solver
            )
            
            if res.success:
                center = res.x[:-1] # Vote distribution
                radius = res.x[-1]  # Robustness radius
                return center, radius
            else:
                return None, 0.0
        except Exception:
            return None, 0.0

def solve_voting_polytope(group, learner=None):
    """
    Constructs the full constraint matrix and solves for the robust vote.
    """
    contestants = group['Contestant'].values
    tech_scores = group['Tech_Score'].values
    n = len(contestants)
    
    # Normalize judge scores for percentage comparison
    total_tech = np.sum(tech_scores) + 1e-9
    tech_share = tech_scores / total_tech
    
    A_ub = []
    b_ub = []
    
    # --- Constraint Set 1: Elimination Logic (Hard) ---
    # Logic: Survivor (S) Total Score >= Eliminated (E) Total Score
    # Implies: v_E - v_S <= t_S - t_E
    elim_indices = np.where(group['Is_Eliminated'] == 1)[0]
    
    if len(elim_indices) > 0:
        # Use the strongest eliminated contestant as the threshold
        elim_idx = elim_indices[np.argmax(tech_scores[elim_indices])]
        
        for idx in range(n):
            if idx in elim_indices: continue
            
            row = np.zeros(n)
            row[elim_idx] = 1.0
            row[idx] = -1.0
            bound = tech_share[idx] - tech_share[elim_idx]
            
            A_ub.append(row)
            b_ub.append(bound)
            
    # --- Constraint Set 2: Network Structure (Soft/Eq.9) ---
    # Logic: If distance d_ij is small, |v_i - v_j| must be small.
    # Implies: v_i - v_j <= delta AND v_j - v_i <= delta
    if learner is not None:
        # Extract features using Q1 model
        feats = learner.get_features(group)
        
        # Apply learned weights if available
        if learner.weights is not None:
            w = np.abs(learner.weights)
            feats = feats * np.sqrt(w) # Weighted feature space
            
        dists = squareform(pdist(feats, metric='euclidean'))
        
        # Slack function: f(d) = base + slope * d
        base_slack = 0.08  # Minimum allowed variance even if identical
        slope = 0.4        # Scaling factor
        
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dists[i, j]
                delta = base_slack + slope * d_ij
                
                # v_i - v_j <= delta
                row_pos = np.zeros(n)
                row_pos[i] = 1.0; row_pos[j] = -1.0
                A_ub.append(row_pos)
                b_ub.append(delta)
                
                # v_j - v_i <= delta
                row_neg = np.zeros(n)
                row_neg[i] = -1.0; row_neg[j] = 1.0
                A_ub.append(row_neg)
                b_ub.append(delta)

    # Convert to arrays
    if len(A_ub) > 0:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        # Dummy constraint if no elimination (e.g., Finals)
        A_ub = np.zeros((1, n))
        b_ub = np.array([1.0])

    # --- Constraint Set 3: Simplex (Equality) ---
    # Sum(v) = 1
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    
    # --- Solve ---
    solver = PolytopeSolver()
    center, radius = solver.find_chebyshev_center(A_ub, b_ub, A_eq, b_eq, n)
    
    # Fallback to uniform if unsolvable (e.g., conflicting constraints)
    if center is None:
        return np.ones(n)/n, 0.0, 1.0 # 1.0 volume implies high uncertainty fallback
        
    # Dimensionality-adjusted robustness score
    robustness_score = radius * np.sqrt(n)
    return center, robustness_score, radius

def analyze_geometric_constraints(df, learner=None):
    """
    Main entry point for geometric analysis.
    Computes Robust_Fan_Vote and Feasible_Volume (via radius proxy).
    """
    print("[Geometric Core] Solving Chebyshev Centers (w/ Network Constraints)...")
    results = []
    
    # Initialize column for robust votes
    df['Robust_Fan_Vote'] = np.nan
    
    # Filter for weeks that actually require solving (usually all)
    groups = [g for _, g in df.groupby(['Season', 'Week'])]
    total = len(groups)
    
    for i, group in enumerate(groups):
        if len(group) < 2: continue
        
        if i % 50 == 0: print(f"  > Processing geometry {i}/{total}...")
        
        # SOLVE
        center, robustness, radius = solve_voting_polytope(group, learner=learner)
        
        # Store reconstructed votes
        df.loc[group.index, 'Robust_Fan_Vote'] = center
        
        # Store metrics
        # Note: We use Radius as a proxy for Volume/Tightness in this precise model
        results.append({
            'Season': group['Season'].iloc[0],
            'Week': group['Week'].iloc[0],
            'Chebyshev_Radius': radius,
            'Feasible_Volume': radius, # For compatibility with VN-APC
            'Robustness_Score': robustness,
            'N_Contestants': len(group)
        })
        
    return pd.DataFrame(results)