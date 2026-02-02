import numpy as np
from scipy.optimize import minimize
import pandas as pd
import networkx as nx 
from scipy.stats import rankdata 

class InversePreferenceLearner:
    def __init__(self):
        self.weights = None 
        # === 升级：特征向量扩展为 8 维 (严格对齐论文计划) ===
        self.feature_names = [
            'Tech',         # 1. 裁判分
            'Pop',          # 2. 知名度
            'Partner',      # 3. 舞伴效应
            'Underdog',     # 4. 弱者/同情分
            'Momentum',     # 5. 动量 (新增)
            'Net_PageRank', # 6. 网络中心性
            'Net_Closeness',# 7. 竞争强度 (Closeness)
            'Net_Cluster'   # 8. 聚类位置 (新增 Clustering Coefficient)
        ]
        
    def _compute_network_metrics(self, df_week):
        """
        [O-Prize Core] 构建动态竞争网络并提取图论特征 (PageRank, Closeness, Clustering)。
        """
        G = nx.Graph()
        contestants = df_week.index.tolist()
        scores = df_week['Tech_Score'].values
        
        G.add_nodes_from(range(len(contestants)))
        
        # 构建全连接加权图
        for i in range(len(contestants)):
            for j in range(i + 1, len(contestants)):
                # 权重: 竞争强度。分数差异越小，强度越大。
                diff = abs(scores[i] - scores[j])
                weight = 1.0 / (diff + 0.05)
                G.add_edge(i, j, weight=weight)
        
        # 1. PageRank: 核心影响力
        try:
            pr = nx.pagerank(G, weight='weight', alpha=0.85)
            pr_vals = [pr[i] for i in range(len(contestants))]
        except:
            pr_vals = [1.0/len(contestants)] * len(contestants)
            
        # 2. Closeness Centrality: 竞争压力/强度
        try:
            cl = nx.closeness_centrality(G, distance=None)
            cl_vals = [cl[i] for i in range(len(contestants))]
        except:
            cl_vals = [0.5] * len(contestants)

        # 3. Clustering Coefficient: 聚类位置 (新增)
        # 衡量选手是否处于一个"高密度竞争集团"中
        try:
            # 使用加权聚类系数
            clust = nx.clustering(G, weight='weight')
            clust_vals = [clust[i] for i in range(len(contestants))]
        except:
            clust_vals = [0.5] * len(contestants)
            
        return np.array(pr_vals), np.array(cl_vals), np.array(clust_vals)

    def get_features(self, df_week):
        # 基础特征
        tech = df_week['Tech_Score'].values
        pop = df_week['Pop_Score'].values
        partner = df_week['Partner_Score'].values
        
        # 特征 4: Underdog (Sigmoid Inversion)
        underdog = 1.0 / (1.0 + np.exp(5.0 * (tech - 0.4)))

        # 特征 5: Momentum (动量 - 新增)
        # 注意: data_loader.py 已经计算了 Momentum_Score，直接提取即可
        if 'Momentum_Score' in df_week.columns:
            momentum = df_week['Momentum_Score'].values
        else:
            momentum = tech # Fallback

        # 网络特征 (Graph Theory)
        pr_vals, cl_vals, clust_vals = self._compute_network_metrics(df_week)
        
        # 归一化辅助函数
        def normalize(v):
            return (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-9)
        
        net_pagerank = normalize(pr_vals)
        net_closeness = normalize(cl_vals)
        net_cluster = normalize(clust_vals) # 新增
        
        # === 堆叠 8 个特征 ===
        return np.column_stack([
            tech, pop, partner, underdog, 
            momentum, net_pagerank, net_closeness, net_cluster
        ])

    def loss_function(self, weights, data_batches):
        w_abs = np.abs(weights)
        w_norm = w_abs / (np.sum(w_abs) + 1e-9)
        loss = 0
        
        RANK_SEASONS = [1, 2] + list(range(28, 35))
        
        entropy = -np.sum(w_norm * np.log(w_norm + 1e-9))
        loss -= 1.5 * entropy  
        
        for batch in data_batches:
            if batch['Is_Eliminated'].sum() == 0: continue
            
            current_season = batch['Season'].iloc[0]
            n_contestants = len(batch)
            
            # 1. 计算预测观众得票率 (Fan Vote Share)
            X = self.get_features(batch)
            utils = np.dot(X, w_norm)
            exp_u = np.exp(utils * 4.0) 
            pred_fan_votes = exp_u / np.sum(exp_u)
            
            # 2. 裁判分数
            tech_scores = batch['Tech_Score'].values
            
            # === 赛季规则区分 ===
            if current_season in RANK_SEASONS:
                # 排名制 (Rank System)
                judge_ranks = rankdata(-tech_scores, method='average')
                fan_ranks = rankdata(-pred_fan_votes, method='average')
                sum_ranks = judge_ranks + fan_ranks
                
                # 逻辑翻转: -SumRank (越小越好 -> 越大越好)
                combined_scores = -1.0 * sum_ranks / n_contestants
                margin = 0.01 
            else:
                # 百分比制 (Percent System)
                judge_share = tech_scores / (np.sum(tech_scores) + 1e-9)
                combined_scores = 0.5 * judge_share + 0.5 * pred_fan_votes
                margin = 0.005 
            
            elim_mask = batch['Is_Eliminated'].values == 1
            if not np.any(elim_mask): continue
            
            score_eliminated = combined_scores[elim_mask]
            score_survivors = combined_scores[~elim_mask]
            
            diffs = score_eliminated.reshape(-1, 1) - score_survivors.reshape(1, -1) + margin
            loss += np.sum(np.maximum(0, diffs)) * 50.0
                
        return loss

    def train_and_predict_walk_forward(self, df):
        print("[Model] Running Walk-Forward Graph Learning (8 Features)...")
        df = df.sort_values(['Season', 'Week'])
        seasons = sorted(df['Season'].unique())
        
        df['Est_Fan_Vote'] = 0.0
        learned_weights_history = []
        
        # 初始化权重 (8个特征)
        # Tech, Pop, Partner, Underdog, Momentum, PR, Closeness, Cluster
        current_weights = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        
        for i, target_season in enumerate(seasons):
            train_seasons = seasons[max(0, i-5):i] 
            if i < 3: train_seasons = seasons[:i]
            
            if train_seasons:
                train_data = df[df['Season'].isin(train_seasons)]
                if not train_data.empty:
                    current_weights = self._fit_batch(train_data, current_weights)
            
            w_norm = np.abs(current_weights) / np.sum(np.abs(current_weights))
            
            record = {'Season_Predicted': target_season}
            for name, w in zip(self.feature_names, w_norm):
                record[name] = w
            learned_weights_history.append(record)
            
            target_data = df[df['Season'] == target_season]
            if not target_data.empty:
                probs_series = self._predict_batch_probs(target_data, w_norm)
                df.loc[target_data.index, 'Est_Fan_Vote'] = probs_series
            
        self.weights = w_norm
        pd.DataFrame(learned_weights_history).to_csv('results/tables/weights_evolution.csv', index=False)
        return df

    def _fit_batch(self, df, init_weights):
        groups = [g for _, g in df.groupby(['Season', 'Week'])]
        import random
        if len(groups) > 150: groups = random.sample(groups, 150)
        
        # === 8特征约束 (Expert Constraints) ===
        bounds = [
            (0.15, 0.60),  # 1. Tech: 技术基石
            (0.10, 0.50),  # 2. Pop: 知名度
            (0.05, 0.40),  # 3. Partner: 舞伴
            (0.01, 0.20),  # 4. Underdog: 严格限制 (<20%)
            (0.01, 0.30),  # 5. Momentum: 动量 (新增) - 适中权重
            (0.01, 0.30),  # 6. PageRank
            (0.01, 0.30),  # 7. Closeness
            (0.01, 0.30)   # 8. Cluster: 聚类位置 (新增)
        ]
        
        # 确保初始权重维度匹配 (以防万一)
        if len(init_weights) != 8:
            init_weights = np.ones(8) / 8.0

        res = minimize(self.loss_function, init_weights, args=(groups,),
                       method='L-BFGS-B', bounds=bounds)
        return res.x

    def _predict_batch_probs(self, df_subset, weights):
        indices = []
        values = []
        for idx, group in df_subset.groupby(['Season', 'Week']):
            X = self.get_features(group)
            utils = np.dot(X, weights)
            exp_u = np.exp(utils * 6.0)
            p = exp_u / np.sum(exp_u)
            indices.extend(group.index.tolist())
            values.extend(p.tolist())
        s = pd.Series(values, index=indices)
        return df_subset.index.map(s)