# ==========================================
# utils/metrics.py
# ==========================================
import torch
import torch.nn.functional as F
import numpy as np

def audit_entropy_share(y_temporal, y_spatial):
    """
    阶段四: 对数信息熵空间的可解释因果份额审计器
    
    物理逻辑：
    由于时间头解冻后存在负数趋势（低于平均温度基数），而空间头恒正（弹性扰动倍数），
    我们在映射至对数能级时，对时间项求取绝对值。这代表“无论趋势向上还是向下，其蕴含的绝对物理能级”
    与空间乘性系数偏离1.0的对数绝对能级（|ln(y_spatial)|）进行线性叠加审计。
    """
    eps = 1e-8
    
    # 1. 时间绝对趋势能级映射：取绝对值后进入对数空间
    ln_y_temporal = torch.log(torch.abs(y_temporal) + eps)
    
    # 2. 空间扰动弹性增益能级映射
    ln_y_spatial = torch.log(y_spatial + eps)
    
    # 3. 提取绝对控制力模长 (能量标量化)
    abs_temporal = torch.abs(ln_y_temporal)
    abs_spatial = torch.abs(ln_y_spatial)
    
    total_energy = abs_temporal + abs_spatial + eps
    
    # 4. 结算全生命周期时空因果博弈份额比例
    w1 = abs_temporal / total_energy
    w2 = abs_spatial / total_energy
    
    return w1.cpu().numpy(), w2.cpu().numpy()


class Evaluator:
    """
    全栈学术评测与诊断报告管理类
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.base_mses, self.base_maes = [], []
        self.prop_mses, self.prop_maes = [], []
        self.all_temporal_weights = []
        self.all_spatial_weights = []
        self.cmi_values = []

    def update_cmi(self, y_true, preds_base, preds_prop, cmi_val, gates):
        """全新 CMI 数据审计拦截器[cite: 9]"""
        self.base_mses.append(F.mse_loss(preds_base, y_true).item())
        self.base_maes.append(F.l1_loss(preds_base, y_true).item())
        self.prop_mses.append(F.mse_loss(preds_prop, y_true).item())
        self.prop_maes.append(F.l1_loss(preds_prop, y_true).item())

        # 拦截并通过对数解耦份额进行全局统计积累
        # 由于 gates 传进来的是计算好的 w1_share (形状为 [B, 24, 7])
        # 我们对其在 Batch 和未来时间、特征轴上取均值，用于宏观文本报告
        w1_mean = gates.mean().item()
        self.all_temporal_weights.append(w1_mean)
        self.all_spatial_weights.append(1.0 - w1_mean) # w1 + w2 = 1 互补
        self.cmi_values.append(cmi_val)

    def report(self):
        mean_base_mse = np.mean(self.base_mses)
        mean_base_mae = np.mean(self.base_maes)
        mean_prop_mse = np.mean(self.prop_mses)
        mean_prop_mae = np.mean(self.prop_maes)
        mean_cmi = np.mean(self.cmi_values)

        print("\n================== 💾 IJCAI 风格 实验结果报告 ==================")
        print(f"1. 预测精度评估 (MSE / MAE):")
        print(f"   [基准黑盒模型 (Baseline)]: MSE = {mean_base_mse:.5f} | MAE = {mean_base_mae:.5f}")
        print(f"   [时空解耦模型 (Proposed)]: MSE = {mean_prop_mse:.5f} | MAE = {mean_prop_mae:.5f}")
        
        print(f"\n2. 内在条件信息论指标 (变分网格估计得到的条件互信息上界，越接近0时空纯度越高):")
        print(f"   [解耦约束模型]: Estimated I(z1; z2 | X) = {mean_cmi:.5f}")
        
        print(f"\n3. 内在自主涌现诊断权重比例 (基于对数信息熵空间无损拆分):")
        print(f"   平均时间周期惯性份额 (w1): {np.mean(self.all_temporal_weights):.2%}")
        print(f"   平均空间跨变量冲击份额 (w2): {np.mean(self.all_spatial_weights):.2%}")
        print("==========================================================")

        # 把收集到的宏观演变动态重新挂载，供主入口 main.py 进行后续的可视化画图
        # 由于可视化脚本需要一维序列，我们直接把每个 Batch 积累的均值拉平提供过去
        self.all_temporal_weights = np.array(self.all_temporal_weights)
        self.all_spatial_weights = np.array(self.all_spatial_weights)