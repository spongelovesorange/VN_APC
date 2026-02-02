import numpy as np
import pandas as pd

def run_vn_apc_simulation(df, topo_df, geo_df):
    print("[System] Running VN-APC Closed-Loop Simulation (Volume-Driven)...")
    
    # --- 1. 建立几何体积映射 (Season, Week -> Volume) ---
    # 修正点：不再寻找 Feasibility_Radius，而是使用 Feasible_Volume
    geo_map = {}
    if not geo_df.empty:
        # 如果 geo_df 中包含 Feasible_Volume (新版)，则使用它
        if 'Feasible_Volume' in geo_df.columns:
            for _, row in geo_df.iterrows():
                geo_map[(row['Season'], row['Week'])] = row['Feasible_Volume']
        # 兼容旧版：如果还是 Radius，则进行转换（防止报错）
        elif 'Feasibility_Radius' in geo_df.columns:
            print("[Warning] Using deprecated 'Feasibility_Radius'. Please update geometric_tools.py.")
            max_r = geo_df['Feasibility_Radius'].max() + 1e-9
            for _, row in geo_df.iterrows():
                geo_map[(row['Season'], row['Week'])] = 1.0 - (row['Feasibility_Radius'] / max_r)
    
    # --- 2. 建立拓扑健康度映射 ---
    topo_map = {}
    if not topo_df.empty:
        for _, row in topo_df.iterrows():
            topo_map[row['Season']] = row.get('Topo_Health', 0.5)

    df['VN_APC_Score'] = 0.0
    df['Adaptive_Alpha'] = 0.5 
    
    for s in df['Season'].unique():
        s_data = df[df['Season'] == s]
        
        # 初始 Alpha 根据赛季拓扑健康度定
        base_health = topo_map.get(s, 0.8)
        current_alpha = 0.5 + (1.0 - base_health) * 0.3
        integral_error = 0.0
        
        for w in sorted(s_data['Week'].unique()):
            group = s_data[s_data['Week'] == w]
            
            # --- 3. Dynamic Setpoint (关键修改) ---
            # 逻辑更新：
            # 可行域体积(geo_vol) 越大 -> 说明约束越松 -> 观众想怎么投怎么投 -> 系统越混乱
            # 因此，当体积大时，我们需要设定更高的相关性目标(Target)来约束系统
            geo_vol = geo_map.get((s, w), 0.5)
            
            # 目标相关性：基础 0.92，体积越大，目标越高
            target_corr = 0.92 + 0.06 * geo_vol 
            target_corr = np.clip(target_corr, 0.90, 0.99)
            
            # --- 4. Feedback ---
            tech = group['Tech_Score'].values
            vote = group['Est_Fan_Vote'].values
            
            if len(tech) > 2 and np.std(tech) > 0:
                current_corr = np.corrcoef(tech, vote)[0, 1]
                if np.isnan(current_corr): current_corr = 0
            else:
                current_corr = 0.8
            
            # --- 5. PID Control ---
            error = target_corr - current_corr
            
            Kp, Ki = 0.8, 0.2
            integral_error = integral_error * 0.8 + error 
            
            delta_alpha = Kp * error + Ki * integral_error
            
            # 加入微小随机扰动，模拟人类决策的非确定性
            wobble = np.random.normal(0, 0.01)
            delta_alpha += wobble
            
            delta_alpha = np.clip(delta_alpha, -0.15, 0.15)
            
            # 限制 Alpha 范围
            current_alpha = np.clip(current_alpha + delta_alpha, 0.2, 0.95)
            
            # --- 6. Actuation ---
            z_tech = (tech - np.mean(tech)) / (np.std(tech) + 1e-9)
            z_vote = (vote - np.mean(vote)) / (np.std(vote) + 1e-9)
            
            final_score = current_alpha * z_tech + (1 - current_alpha) * z_vote
            
            df.loc[group.index, 'VN_APC_Score'] = final_score
            df.loc[group.index, 'Adaptive_Alpha'] = current_alpha
            
    return df