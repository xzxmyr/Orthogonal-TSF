# ==========================================
# utils/visualization.py
# ==========================================
import os
import matplotlib
# 在服务器后台静默绘图，禁止弹出GUI窗口，防范 Display 报错
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_and_save_weights(temporal_weights, spatial_weights=None, save_path="weight_dynamics.png", window_size=50):
    """
    绘制动态权重解耦谱图 (Dynamic Weight Allocation Spectrum)
    
    物理机制：
    利用门控机制 w1 + w2 = 1 的公理化特征，绘制单折线边界并填充上下双色堆叠面积。
    图形上半部分（蓝色）代表跨变量空间耦合权重的贡献，下半部分（红色）代表纯时间周期规律权重的贡献。
    
    :param temporal_weights: 时间权重 w1 序列，支持 list, numpy 数组或 PyTorch Tensor
    :param spatial_weights: 空间权重 w2 序列（可选）。由于 w1+w2=1，如果不传则自动逆向推导
    :param save_path: 谱图保存的本地目标路径
    :param window_size: 滑动平均窗口大小，用以平滑测试集宏观演变趋势，消除微观毛刺
    """
    # 1. 健壮性规整：确保输入转为一维 NumPy 数组
    if hasattr(temporal_weights, 'cpu'):
        w1 = temporal_weights.cpu().numpy()
    else:
        w1 = np.array(temporal_weights)
    
    # 彻底解开高维潜流可能残存的 Batch 或 Time 冗余单维度 [N, 1] -> [N]
    w1 = w1.squeeze()
    if w1.ndim > 1:
        w1 = w1.flatten()

    # 2. 宏观动力学趋势平滑 (滑动平均优化)
    if len(w1) > window_size:
        w1_smooth = pd.Series(w1).rolling(window=window_size, min_periods=1).mean().values
    else:
        w1_smooth = w1
        
    x_axis = np.arange(len(w1_smooth))
    
    # 3. 初始化符合顶级期刊规范的高分辨率画布 (300 DPI)
    plt.figure(figsize=(12, 5.5), dpi=300)
    
    # 4. 绘制核心分割边界线 (代表 w1 的解耦动态分水岭)
    # 采用高冷优雅的深灰黑 (#2C3E50)
    plt.plot(x_axis, w1_smooth, color='#2C3E50', lw=1.8, linestyle='-', label='Decoupling Boundary ($w_1$)')
    
    # 5. 双色互补流形填充
    # 上方填充 w2 蔚蓝流形 (#3498DB) -> 从 w1_smooth 边界一直覆盖到天花板 y=1.0
    plt.fill_between(x_axis, w1_smooth, 1.0, color='#3498DB', alpha=0.75, 
                     label='Variable Coupling Weight ($w_2$)')
    
    # 下方填充 w1 朱红流形 (#E74C3C) -> 从地板 y=0.0 一直覆盖到 w1_smooth 边界
    plt.fill_between(x_axis, 0.0, w1_smooth, color='#E74C3C', alpha=0.75, 
                     label='Temporal Pattern Weight ($w_1$)')
    
    # 6. 学术视觉美化与边界严限制
    plt.xlim(x_axis[0], x_axis[-1])
    plt.ylim(0.0, 1.0)  # 概率空间严格控制在 0 到 1 之间
    
    # 设置学术英文标准 Label
    plt.xlabel("Test Set Observations (Chronological Samples)", fontsize=11, labelpad=8)
    plt.ylabel("Weight Contribution Proportion", fontsize=11, labelpad=8)
    plt.title("Dynamic Weight Allocation Spectrum ($w_1 + w_2 = 1$)", fontsize=13, fontweight='bold', pad=15)
    
    # 7. 现代平面设计感美化：去粗取精
    # 保留轻量纵向网格作为时间刻度辅助参照
    plt.grid(axis='x', color='gray', linestyle='--', alpha=0.25)
    # 精简框线：擦除顶部和右侧冗余边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 8. 图例浮动阴影与去边框优化
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=10)
    
    # 9. 智能剔除版面死白空间
    plt.tight_layout()
    
    # 10. 安全建立父级目录并持久化存储
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    plt.savefig(save_path)
    plt.close()
    print(f"🎉 完美的双色互补权重动态谱图已成功保存至: {save_path}")