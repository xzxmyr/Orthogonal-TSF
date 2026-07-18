import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangledProposedModel(nn.Module):
    """
    改进模型：拆分出纯时间规律 z1、纯跨变量空间 z2，并用门控机制融合，输出实时权重
    """
    def __init__(self, seq_len, num_features, pred_len, hidden_dim=64):
        super(DisentangledProposedModel, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.pred_len = pred_len
        
        # 时间表征分支 (Intra-series): 处理局部时间依赖关系
        self.temporal_encoder = nn.GRU(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
        
        # 空间/跨变量分支 (Inter-series): 使用一维卷积/线性层处理同一时间步不同变量的映射
        self.spatial_encoder = nn.Linear(num_features, hidden_dim)
        
        # 门控网络: 依据合并特征判断当前预测应偏向时间(w1)还是空间(w2)
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1) # w1 + w2 = 1
        )
        
        self.predictor = nn.Linear(hidden_dim, num_features * pred_len)

    def forward(self, x):
        B, L, F = x.shape
        
        # 1. 提取时间特征 z1
        z1, _ = self.temporal_encoder(x) #
        
        # 2. 提取空间特征 z2
        z2 = self.spatial_encoder(x) #
        
        # 3. 动态门控融合权重计算
        z_concat = torch.cat([z1, z2], dim=-1) #
        z_mean = z_concat.mean(dim=1) # 全局序列特征池化
        
        gate_weights = self.gating_network(z_mean) # -> w1, w2
        w1 = gate_weights[:, 0].view(B, 1, 1) # 时间权重
        w2 = gate_weights[:, 1].view(B, 1, 1) # 变量耦合权重
        
        # 4. 融合后的表征 z_fused
        z_fused = w1 * z1 + w2 * z2 #
        z_fused_last = z_fused[:, -1, :] #
        
        preds = self.predictor(z_fused_last).view(B, self.pred_len, F)
        
        return preds, z1, z2, gate_weights
