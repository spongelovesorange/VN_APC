import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_env():
    """
    配置出版级绘图环境 (Publication Quality Plotting)
    """
    # 使用 Seaborn 的 Paper 风格，字体清晰
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    # 专业配色：蓝色(裁判/客观), 红色(观众/主观), 紫色(控制), 灰色(辅助)
    custom_palette = ["#348ABD", "#E24A33", "#988ED5", "#777777"]
    sns.set_palette(custom_palette)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 自动创建结果目录
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)
    print("[Utils] Environment setup complete.")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_figure(filename, dpi=300):
    """保存高分辨率图片"""
    path = os.path.join('results/figures', filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"[Utils] Saved figure: {path}")

def stable_softmax(x, temperature=1.0):
    """数值稳定的 Softmax"""
    x_shifted = (x - np.max(x)) / temperature
    exps = np.exp(x_shifted)
    return exps / np.sum(exps)