import pandas as pd
import numpy as np
import re
import os

def load_fan_data(fan_path):
    """
    读取社交媒体粉丝数据，并构建 {姓名: 归一化分数} 的映射字典
    """
    if not os.path.exists(fan_path):
        print(f"[Warning] 找不到粉丝数据文件: {fan_path}，将完全使用行业规则。")
        return {}

    try:
        df = pd.read_csv(fan_path)
        # 智能查找列名
        name_col = next((c for c in df.columns if 'name' in c.lower() or 'celebrity' in c.lower()), df.columns[0])
        fans_col = next((c for c in df.columns if 'fan' in c.lower() or 'follow' in c.lower() or 'count' in c.lower()), df.columns[1])
        
        print(f"[Data] 粉丝数据加载成功。姓名列: {name_col}, 粉丝数列: {fans_col}")
        
        # 清洗数据
        fan_map = {}
        raw_values = []
        
        for _, row in df.iterrows():
            name = str(row[name_col]).strip().lower()
            try:
                # 处理 "1.2M", "500k" 或纯数字格式
                val_str = str(row[fans_col]).lower().replace(',', '')
                if 'm' in val_str:
                    val = float(val_str.replace('m', '')) * 1_000_000
                elif 'k' in val_str:
                    val = float(val_str.replace('k', '')) * 1_000
                else:
                    val = float(val_str)
                
                # 取对数处理，防止头部效应过大
                log_val = np.log10(val + 1)
                fan_map[name] = log_val
                raw_values.append(log_val)
            except:
                continue
                
        # 全局归一化 (Min-Max) 到 0.2 ~ 1.0
        if raw_values:
            v_min, v_max = min(raw_values), max(raw_values)
            for k in fan_map:
                norm_score = 0.2 + 0.8 * (fan_map[k] - v_min) / (v_max - v_min + 1e-9)
                fan_map[k] = norm_score
                
        return fan_map

    except Exception as e:
        print(f"[Warning] 粉丝数据解析失败: {e}")
        return {}

def load_and_clean_data(filepath, fan_filepath=None):
    """
    主数据加载函数：提取所有关键特征（包括年龄、行业、动量、舞伴等）
    """
    print(f"[Data] Loading main data from {filepath}...")
    
    # 1. 加载粉丝字典
    fan_dict = {}
    if fan_filepath:
        fan_dict = load_fan_data(fan_filepath)
        print(f"[Data] 已导入 {len(fan_dict)} 位明星的真实粉丝数据。")

    # 2. 读取主数据文件
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[Fatal] 找不到数据文件: {filepath}")

    try:
        raw_df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except:
        raw_df = pd.read_csv(filepath, encoding='utf-8')

    # 标准化列名
    raw_df.columns = [str(c).strip().lower().replace('\\n', ' ').replace(' ', '_') for c in raw_df.columns]
    
    # 智能寻找关键列
    name_col = next((c for c in raw_df.columns if 'celebrity' in c and 'name' in c), None)
    if not name_col: name_col = next((c for c in raw_df.columns if 'celebrity' in c), None)
    if not name_col: name_col = next((c for c in raw_df.columns if 'contestant' in c), None)
    
    # === 关键修复：寻找 Age 列 ===
    age_col = next((c for c in raw_df.columns if 'age' in c), None)
    
    # === 关键修复：寻找 Industry 列 ===
    ind_col = next((c for c in raw_df.columns if 'industry' in c), None)
    
    score_cols = [c for c in raw_df.columns if 'judge' in c and 'score' in c]
    res_col = next((c for c in raw_df.columns if 'result' in c or 'place' in c), None)

    data_list = []
    
    for idx, row in raw_df.iterrows():
        # 提取赛季
        season = row.get('season', 0)
        try: season = int(season)
        except: season = 0
            
        # 提取名字
        contestant_raw = str(row.get(name_col, f"Unknown_{idx}")).strip()
        contestant_key = contestant_raw.lower() 
        
        # 提取舞伴
        partner = str(row.get('ballroom_partner', 'Unknown')).strip()
        
        # === 提取行业 (保留原始文本) ===
        industry_raw = str(row.get(ind_col, 'Other')).strip() if ind_col else 'Other'
        
        # === 提取年龄 (处理空值) ===
        age_val = np.nan
        if age_col:
            try:
                raw_age = str(row.get(age_col, '')).replace('nan', '').strip()
                if raw_age:
                    age_val = float(raw_age)
            except:
                age_val = np.nan
        
        # === 知名度 (Pop_Score) 计算逻辑 ===
        # 优先级: 真实数据 > 行业规则 heuristic
        if contestant_key in fan_dict:
            base_pop = fan_dict[contestant_key]
        else:
            ind_lower = industry_raw.lower()
            if any(x in ind_lower for x in ['reality', 'kardashian', 'bachelor', 'youtube', 'influencer']): 
                base_pop = 0.85
            elif any(x in ind_lower for x in ['athlete', 'nba', 'nfl', 'olympi', 'football']): 
                base_pop = 0.80
            elif 'singer' in ind_lower or 'pop' in ind_lower or 'music' in ind_lower: 
                base_pop = 0.70
            elif 'actor' in ind_lower or 'actress' in ind_lower:
                base_pop = 0.60
            else:
                base_pop = 0.40
        
        # === 确定淘汰周 ===
        exit_week_text = 99
        if res_col:
            res_str = str(row.get(res_col, '')).lower()
            if 'eliminated' in res_str:
                match = re.search(r'week\s*(\d+)', res_str)
                if match: exit_week_text = int(match.group(1))
            elif any(x in res_str for x in ['winner', 'runner', 'place', 'final']):
                exit_week_text = 99
        
        # === 解析每周分数并展平数据 ===
        last_active_week = 0
        week_cols_map = {} 
        for c in score_cols:
            m = re.search(r'week(\d+)', c)
            if m:
                w = int(m.group(1))
                if w not in week_cols_map: week_cols_map[w] = []
                week_cols_map[w].append(c)
                val = pd.to_numeric(row[c], errors='coerce')
                if pd.notna(val) and val > 0:
                    if w > last_active_week: last_active_week = w
        
        final_exit_week = 99 if exit_week_text == 99 else last_active_week
        
        cumulative_tech = 0
        current_pop = base_pop
        
        for w in sorted(week_cols_map.keys()):
            # 过滤掉已经淘汰后的周次
            if w > final_exit_week and final_exit_week != 99: continue
                
            vals = [pd.to_numeric(row[c], errors='coerce') for c in week_cols_map[w]]
            vals = [v for v in vals if pd.notna(v) and v > 0]
            if not vals: continue 
            
            # 计算技术分
            avg_score = np.mean(vals)
            max_score = 40 if avg_score > 30 else 30 
            tech_score = avg_score / max_score
            
            # 计算动量 (Momentum)
            cumulative_tech += tech_score
            momentum = cumulative_tech / w 
            
            # Pop 随机游走 (模拟人气波动)
            current_pop += np.random.normal(0, 0.01)
            current_pop = np.clip(current_pop, 0.1, 1.0)
            
            # 标记是否本周淘汰
            is_elim = 1 if (final_exit_week != 99 and w == final_exit_week) else 0
            
            data_list.append({
                'Season': season,
                'Week': w,
                'Contestant': contestant_raw,
                'Partner': partner,
                'Industry_Raw': industry_raw, # 关键：保留原始行业
                'Age': age_val,               # 关键：保留年龄
                'Tech_Score': tech_score,
                'Pop_Score': current_pop,
                'Momentum_Score': momentum,
                'Is_Eliminated': is_elim
            })

    df_clean = pd.DataFrame(data_list)
    
    # === 计算 Partner_Score (舞伴能力值) ===
    # 基于该舞伴历史平均技术分
    partner_stats = df_clean.groupby('Partner')['Tech_Score'].mean().to_dict()
    df_clean['Partner_Score'] = df_clean['Partner'].map(partner_stats)
    
    # 归一化舞伴分数
    p_min, p_max = df_clean['Partner_Score'].min(), df_clean['Partner_Score'].max()
    if p_max > p_min:
        df_clean['Partner_Score'] = (df_clean['Partner_Score'] - p_min) / (p_max - p_min)
    else:
        df_clean['Partner_Score'] = 0.5 

    return df_clean