# ==========================================
# losses/info_losses.py
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalLoss(nn.Module):
    """
    几何正交去相关软约束损失函数 (Orthogonal Regularization Loss)
    
    物理意义：
    强制将时间特征流流形 z1 与空间特征流流形 z2 在高维隐空间内推至严格垂直（90°），
    从而在数学层面上斩断两者的线性共线性，彻底绝禁前向传播中的“表征泄漏（Representation Leakage）”。
    """
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, z1, z2):
        """
        前向传播计算正交惩罚项
        :param z1: 时间分支隐表征, 形状通常为 [B, L, H] (Batch, Length, Hidden_Dim)
        :param z2: 空间分支隐表征, 形状与 z1 严格一致 [B, L, H]
        :return: 标量标量损失值 loss_ort
        """
        # 学术安全检查：确保输入的时空潜在特征张量形状完全一致
        assert z1.shape == z2.shape, f"时空表征维度不匹配! z1 形状: {z1.shape}, z2 形状: {z2.shape}"
        
        # 1. 在隐空间特征维度 H (最后一个维度 dim=-1) 上进行严格的 L2 归一化
        # eps=1e-8 防止分母为 0 导致梯度出现 NaN
        z1_norm = F.normalize(z1, p=2, dim=-1, eps=1e-8)
        z2_norm = F.normalize(z2, p=2, dim=-1, eps=1e-8)
        
        # 2. 计算在每个 Batch 样本的每一个时间步上的点积（即余弦相似度）
        # 形状变化: [B, L, H] * [B, L, H] -> sum(dim=-1) -> [B, L]
        cos_sim = (z1_norm * z2_norm).sum(dim=-1)
        
        # 3. 惩罚其余弦相似度的平方
        # 物理逻辑：其余弦值可能为正（夹角 < 90°）或负（夹角 > 90°），平方操作确保任意方向的靠拢都会触发巨大惩罚，
        # 进而逼迫系统将 cos_sim 压向 0（即严格正交）。
        loss_ort = torch.mean(cos_sim ** 2)
        
        return loss_ort