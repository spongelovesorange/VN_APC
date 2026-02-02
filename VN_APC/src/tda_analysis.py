import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import warnings

# Suppress TDA library warnings for cleaner logs
warnings.filterwarnings("ignore")

try:
    from ripser import ripser
    from persim import plot_diagrams
except ImportError:
    print("[Error] 'ripser' or 'persim' not installed. TDA functionality limited.")

def calculate_hodgerank_metrics(pivot_df):
    """
    Performs HodgeRank decomposition on the pairwise preference matrix.
    Decomposes preference matrix Y into Gradient (consistent) and Curl (cyclic) components.
    
    Mathematical Formulation:
    Minimize || (s_i - s_j) - Y_ij ||^2  =>  L * s = div(Y)
    Discrepancy = || Y_curl || / || Y ||
    """
    contestants = pivot_df.columns
    n = len(contestants)
    if n < 3: return 0.0, 0.0

    # 1. Construct Pairwise Preference Matrix Y (Skew-symmetric)
    values = pivot_df.values
    Y = np.zeros((n, n))
    
    # Vectorized construction of Y
    # Y_ij represents the aggregate preference strength of i over j
    for t in range(values.shape[0]):
        row = values[t, :]
        diff = row[:, None] - row[None, :]
        Y += np.tanh(diff * 3.0) # Softmax-like activation for preference intensity
    
    Y = Y / (values.shape[0] + 1e-9)

    # 2. Construct Graph Laplacian (L) and Divergence (div)
    # For a complete graph comparison topology
    A = np.ones((n, n)) - np.eye(n)
    D = np.diag(np.sum(A, axis=1))
    L = D - A 
    
    # Divergence: net flow out of each node
    div_Y = np.sum(Y, axis=1)

    # 3. Solve L * s = div_Y for global ranking scores s (Gradient Potential)
    # Using Least Squares for numerical stability on singular L
    try:
        solution = lsqr(L, div_Y)
        s = solution[0]
    except Exception:
        return 0.0, 1.0

    # 4. Reconstruct Gradient Flow and Extract Curl
    Y_grad = s[:, None] - s[None, :]
    Y_curl = Y - Y_grad # Residual is the cyclic component

    # 5. Calculate Energy Metrics
    energy_total = np.linalg.norm(Y) ** 2
    energy_curl = np.linalg.norm(Y_curl) ** 2
    
    if energy_total < 1e-9: return 0.0, 0.0
    
    discrepancy = np.sqrt(energy_curl / energy_total)
    cyclic_energy = energy_curl
    
    return discrepancy, cyclic_energy

def analyze_season_topology(df):
    """
    Executes dual-layer topological analysis:
    1. Persistent Homology (Betti-1 loops via Ripser)
    2. HodgeRank Decomposition (Cyclic inconsistency via Helmholtz decomposition)
    """
    print("[System] Running Topological Data Analysis (PH + HodgeRank)...")
    results = []
    os.makedirs('results/figures/tda', exist_ok=True)
    
    # Normalize votes for metric space construction
    df = df.copy()
    df['Norm_Vote'] = df.groupby(['Season', 'Week'])['Est_Fan_Vote'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    for s in df['Season'].unique():
        season_data = df[df['Season'] == s]
        pivot = season_data.pivot_table(
            index='Week', columns='Contestant', values='Norm_Vote'
        ).fillna(0)
        
        if pivot.shape[1] < 3:
            results.append({'Season': s, 'Betti_1': 0, 'Hodge_Disc': 0.0, 'Topo_Health': 1.0})
            continue

        # --- Metric 1: HodgeRank Discrepancy ---
        hodge_disc, cyclic_energy = calculate_hodgerank_metrics(pivot)

        # --- Metric 2: Persistent Homology (Betti-1) ---
        betti_1 = 0
        persistence_score = 0.0
        
        try:
            # Construct correlation distance metric: d = sqrt(2(1-rho))
            # Strict Euclidean mapping for TDA
            corr = pivot.corr().fillna(0).values
            dist_matrix = np.sqrt(2 * (1.0 - corr))
            np.fill_diagonal(dist_matrix, 0)

            # Compute Persistence Diagrams (H1)
            tda_res = ripser(dist_matrix, distance_matrix=True, maxdim=1)
            dgms = tda_res['dgms']

            if len(dgms) > 1 and len(dgms[1]) > 0:
                # Lifetime = Death - Birth
                lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
                # Filter noise (short-lived loops)
                significant = lifetimes[lifetimes > 0.15] 
                betti_1 = len(significant)
                persistence_score = np.sum(significant)
                
                # Save diagram for significant anomalies
                if betti_1 > 0:
                    plt.figure(figsize=(5, 4))
                    plot_diagrams(dgms, show=False)
                    plt.title(f"Season {s}: Betti-1={betti_1}, HodgeDisc={hodge_disc:.2f}")
                    plt.tight_layout()
                    plt.savefig(f'results/figures/tda/season_{s}_topology.png', dpi=150)
                    plt.close()

        except Exception as e:
            print(f"[Warning] TDA failed for S{s}: {e}")

        # Composite Topological Health Score
        # Inverse sigmoid of combined inconsistencies
        penalty = 0.7 * persistence_score + 2.0 * hodge_disc
        health = 1.0 / (1.0 + penalty)

        results.append({
            'Season': s,
            'Betti_1': betti_1,
            'Hodge_Disc': hodge_disc,
            'Cyclic_Energy': cyclic_energy,
            'H1_Persistence': persistence_score,
            'Topo_Health': health
        })

    return pd.DataFrame(results)