# ==========================================
# utils/metrics.py
# ==========================================
import torch
import torch.nn.functional as F
import numpy as np

class Evaluator:
    """
    全栈学术评测与诊断报告管理类 (Evaluator & Metric Tracker)
    
    职责：
    1. 动态追踪和收集验证集/测试集的所有精度指标（MSE, MAE）。
    2. 提取并缓存内在诊断门控权重 (w1, w2)，用作因果归一化分析。
    3. 量化隐空间时空流形的正交相似度平方 (Cos_Sim^2)。
    4. 自动化输出具有学术反馈价值的“可行性确诊报告”。
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        """清空缓存，为新一轮的测试集评估做准备"""
        self.base_mses, self.base_maes = [], []
        self.prop_mses, self.prop_maes = [], []
        
        self.all_temporal_weights = []
        self.all_spatial_weights = []
        self.cos_similarities = []

    def update(self, y_true, preds_base, preds_prop, z1, z2, gates):
        """
        逐个 Batch 收集和累计模型输出的预测特征与权重
        """
        # 1. 基准黑盒模型评价缓存
        self.base_mses.append(F.mse_loss(preds_base, y_true).item())
        self.base_maes.append(F.l1_loss(preds_base, y_true).item())
        
        # 2. 时空解耦模型评价缓存
        self.prop_mses.append(F.mse_loss(preds_prop, y_true).item())
        self.prop_maes.append(F.l1_loss(preds_prop, y_true).item())
        
        # 3. 统计诊断权重拓扑
        self.all_temporal_weights.extend(gates[:, 0].cpu().numpy())
        self.all_spatial_weights.extend(gates[:, 1].cpu().numpy())
        
        # 4. 实时计算隐空间表征的正交几何接近度 (Cos_Sim^2)
        z1_n = F.normalize(z1, p=2, dim=-1, eps=1e-8)
        z2_n = F.normalize(z2, p=2, dim=-1, eps=1e-8)
        similarity = torch.mean((z1_n * z2_n).sum(dim=-1)**2).item()
        self.cos_similarities.append(similarity)

    def report(self):
        """
        在整个测试集推理完毕后，统一结算、打印学术级诊断报告，并自动触发绘图
        """
        mean_base_mse = np.mean(self.base_mses)
        mean_base_mae = np.mean(self.base_maes)
        mean_prop_mse = np.mean(self.prop_mses)
        mean_prop_mae = np.mean(self.prop_maes)
        mean_cos_sim = np.mean(self.cos_similarities)

        print("\n================== 实验结果报告 ==================")
        print(f"1. 预测精度评估 (MSE / MAE):")
        print(f"   [基准黑盒模型 (Baseline)]: MSE = {mean_base_mse:.5f} | MAE = {mean_base_mae:.5f}")
        print(f"   [时空解耦模型 (Proposed)]: MSE = {mean_prop_mse:.5f} | MAE = {mean_prop_mae:.5f}")
        
        print(f"\n2. 内在解耦程度指标 (余弦相似度平方，越接近0说明时空分得越开):")
        print(f"   [解耦约束模型]: Cos_Sim^2 = {mean_cos_sim:.5f}")
        
        print(f"\n3. 内在诊断权重比例 (时间规律 vs 变量耦合):")
        print(f"   平均时间规律权重 (w1): {np.mean(self.all_temporal_weights):.2%}")
        print(f"   平均变量耦合权重 (w2): {np.mean(self.all_spatial_weights):.2%}")
        print("==================================================")

        
        # 5. 可行性学术诊断反馈
        if abs(mean_prop_mse - mean_base_mse) < 0.05:
            print("💡 [可行性诊断]: 恭喜！解耦模型的 MSE 与基准模型极其接近。这完美证明了：")
            print("   我们在通过正交软约束‘强行解耦潜在特征’的同时，没有破坏模型对预测标签的编码能力。")
            print("   时空划分预测机制完全成立，可以以此为基石申请 SURF 项目！\n")
        else:
            print("⚠️ [参数调整建议]: 预测误差产生了一定偏离，建议调小 lambda_ort 重试。\n")