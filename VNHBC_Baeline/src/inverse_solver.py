import numpy as np
from scipy.stats import rankdata

class InverseVoter:
    """
    Implements the Inverse Optimization logic using Metropolis-Hastings MCMC.
    
    Key Features:
    1. Zipfian Prior: Enforces the 'Star Power' distribution (Power Law).
    2. Time Continuity: Uses the previous week's posterior as the current week's prior.
    3. Constraint Satisfaction: Ensures estimated votes align with historical elimination results.
    """

    def __init__(self, n_samples=3000, burn_in=500, alpha_zipf=1.1, momentum_weight=15.0):
        """
        Initialize the MCMC Solver.

        Args:
            n_samples (int): Number of MCMC iterations after burn-in.
            burn_in (int): Number of initial iterations to discard.
            alpha_zipf (float): The exponent for the Zipfian distribution (measure of inequality).
            momentum_weight (float): Strength of the time continuity prior. 
                                     Higher values force the model to stick closer to last week's results.
        """
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.alpha_zipf = alpha_zipf
        self.momentum_weight = momentum_weight

    def log_prior(self, votes, prior_means=None):
        """
        Calculate the Log-Prior Probability of a given vote distribution.
        
        Combination of:
        1. Structural Prior (Zipf): Popularity is rarely uniform.
        2. Historical Prior (Dirichlet): Momentum from previous weeks or industry baselines.
        """
        # --- Component 1: Zipfian Structural Prior ---
        # Ranks follow a Power Law: P(r) ~ r^(-alpha)
        # We define the prior on the *ranks* of the vote vector.
        ranks = rankdata(-votes, method='ordinal') 
        log_p_zipf = -self.alpha_zipf * np.sum(np.log(ranks))
        
        # --- Component 2: Informative Historical Prior (Dirichlet) ---
        if prior_means is not None:
            # Clip to avoid numerical instability with log(0)
            safe_means = np.clip(prior_means, 1e-6, 1.0)
            
            # Normalize to ensure it sums to 1 (valid probability simplex)
            safe_means /= np.sum(safe_means)
            
            # Construct Dirichlet Concentration Parameters (Alpha Vector)
            # alpha = weight * mean_probability
            alpha_dir = self.momentum_weight * safe_means
            
            # Calculate Dirichlet Log-PDF (Simplified proportional term)
            # Sum( (alpha_i - 1) * log(x_i) )
            log_p_history = np.sum((alpha_dir - 1) * np.log(votes + 1e-9))
        else:
            log_p_history = 0.0
            
        return log_p_zipf + log_p_history

    def check_constraint(self, judge_scores, votes, is_eliminated, rule_type):
        """
        Verify if a proposed vote vector satisfies the historical elimination constraints.
        
        The eliminated contestant MUST have a lower total score (or higher rank sum) 
        than all surviving contestants.
        """
        if sum(is_eliminated) == 0:
            return True

        # Calculate Total Scores based on the Season's Rule
        if rule_type == 'percent':
            # Percentage Based System (Seasons 3-27)
            # Score = 50% Judge Share + 50% Vote Share
            j_share = judge_scores / (np.sum(judge_scores) + 1e-9)
            total_score = 0.5 * j_share + 0.5 * votes
            
            # Logic: Higher score is better.
            # Constraint: Max(Eliminated) < Min(Survivors)
            
        else: 
            # Rank Based System (Seasons 1-2, 28+)
            # Score = Judge Rank + Vote Rank
            # Note: scipy.stats.rankdata assigns 1 to smallest, so we rank (-score).
            j_rank = rankdata(-judge_scores, method='min')
            v_rank = rankdata(-votes, method='min')
            total_score = -1 * (j_rank + v_rank) 
            # Logic: We use negative rank so that "Higher" number is still "Better/Safer".
            # (e.g., Rank 1 becomes -1, Rank 10 becomes -10. -1 > -10).

        elim_indices = [i for i, x in enumerate(is_eliminated) if x == 1]
        surv_indices = [i for i, x in enumerate(is_eliminated) if x == 0]
        
        if not elim_indices or not surv_indices:
            return True

        # The best score among those eliminated must be worse than the worst score among survivors
        # to justify why *they* went home and not the survivor.
        min_survivor_score = np.min(total_score[surv_indices])
        max_elim_score = np.max(total_score[elim_indices])
        
        # Returns True if the scenario is physically possible
        return max_elim_score <= min_survivor_score

    def solve(self, judge_scores, is_eliminated, rule_type, prior_means=None):
        """
        Execute the MCMC sampling to find the most likely vote distribution.
        
        Returns:
            estimated_votes (np.array): The mean of the sampled posterior distribution.
            is_anomaly (bool): True if no feasible solution was found (Shock Elimination).
        """
        n = len(judge_scores)
        
        # --- Initialization ---
        # If we have prior knowledge (history), start sampling near that region to speed up convergence.
        if prior_means is not None:
            init_alpha = self.momentum_weight * np.clip(prior_means, 0.01, 1.0)
            current_votes = np.random.dirichlet(init_alpha)
        else:
            current_votes = np.random.dirichlet(np.ones(n))
            
        samples = []
        
        # --- Metropolis-Hastings Loop ---
        for i in range(self.n_samples + self.burn_in):
            # 1. Proposal Step: Random Walk on the Simplex
            noise = np.random.normal(0, 0.05, n) # Step size 0.05
            proposal = np.abs(current_votes + noise)
            proposal /= np.sum(proposal) # Re-normalize
            
            # 2. Hard Constraint Check (Likelihood = 0 or 1)
            if not self.check_constraint(judge_scores, proposal, is_eliminated, rule_type):
                continue # Reject immediately if historical reality is violated
            
            # 3. Acceptance Ratio (Prior Probability)
            lp_current = self.log_prior(current_votes, prior_means)
            lp_proposal = self.log_prior(proposal, prior_means)
            acceptance_prob = np.exp(lp_proposal - lp_current)
            
            # 4. Accept/Reject Decision
            if np.random.rand() < acceptance_prob:
                current_votes = proposal
                if i >= self.burn_in:
                    samples.append(current_votes)
        
        # --- Result Aggregation ---
        if not samples:
            # Convergence Failure: The constraints formed an empty set (or we couldn't find it).
            # This indicates a "Shock Elimination" where the model fails to explain reality.
            # Return a random vector and flag as anomaly.
            return np.random.dirichlet(np.ones(n)), True 
            
        return np.mean(samples, axis=0), False