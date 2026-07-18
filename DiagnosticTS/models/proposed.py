import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagnosticTS(nn.Module):
    """
    [内生残差替换与高阶残差微调升级版] 核心可解释时序模型: DiagnosticTS
    1. 强力补全阶段一的 R-Substitution 机制，纯化高频异常相变。
    2. 引入非线性高阶解码与残差校正头，突破传统乘性融合的线性限制。
    """
    def __init__(self, seq_len=96, num_features=7, pred_len=24, hidden_dim=64):
        super(DiagnosticTS, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # ==== 阶段一基准趋势外推头 (Base Trend Extrapolator) ====
        self.base_predictor = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        
        # ==== 阶段二各向异性时空特征提取算子 ====
        self.temporal_encoder = nn.GRU(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(seq_len * num_features, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # ==== 阶段三双向物理预测头 ====
        # 时间趋势头：全量纲释放，允许输出负值基数以解除精度封印
        self.temporal_head = nn.Linear(hidden_dim, num_features * pred_len)
        # 空间增益头：保持恒正，充当环境扰动弹性系数
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, num_features * pred_len),
            nn.Softplus()
        )
        
        # ==== 阶段五硬核拓展：高阶非线性残差微调校正头 (Residual Fine-tuning Head) ====
        # 融合后的物理骨架由该分支进行加性微调，捕获高阶非线性突变动力学
        self.residual_correction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features * pred_len)
        )

    def forward(self, x_hist):
        """
        x_hist: [B, 96, 7] 原始滑动历史窗口
        """
        B, L, F_dim = x_hist.shape
        
        # ────────────────────────────────────────────────────────
        # 阶段一：一阶残差空间注入与流形重构 (R-Substitution)
        # ────────────────────────────────────────────────────────
        x_target = x_hist[:, :, 0] # 提取第 0 通道目标变量（OT）
        x_base_hat = self.base_predictor(x_target) # 基准趋势外推
        r_hist = x_target - x_base_hat # 计算历史一阶残差
        
        # 保持其他 6 个物理传感器通道不动，用 r_hist 强行替换第 0 通道
        x_new_hybrid = torch.cat([r_hist.unsqueeze(-1), x_hist[:, :, 1:]], dim=-1)
        
        # ────────────────────────────────────────────────────────
        # 阶段二：全新流形上的时空算子分裂提取
        # ────────────────────────────────────────────────────────
        # 时间头滚动提取
        _, h_n = self.temporal_encoder(x_new_hybrid)
        z1_raw = h_n[-1] # [B, hidden_dim]
        
        # 空间头平铺横向打碎连续性
        x_flatten = x_new_hybrid.reshape(B, -1)
        z2_raw = self.spatial_encoder(x_flatten) # [B, hidden_dim]
        
        # 【物理非负看守】：大门口防范对抗混乱，雕刻出正定隐空间流形
        z1 = F.softplus(z1_raw)
        z2 = F.softplus(z2_raw)
        
        # ────────────────────────────────────────────────────────
        # 阶段三：物理空间各项同性哈达玛乘性融合
        # ────────────────────────────────────────────────────────
        y_temporal = self.temporal_head(z1).view(B, self.pred_len, F_dim)
        y_spatial = self.spatial_head(z2).view(B, self.pred_len, F_dim)
        
        y_skeleton = y_temporal * y_spatial # 乘性主干物理骨架
        
        # ────────────────────────────────────────────────────────
        # 拓展微调：高阶非线性残差校正
        # ────────────────────────────────────────────────────────
        z_concat = torch.cat([z1, z2], dim=-1)
        y_residual = self.residual_correction_head(z_concat).view(B, self.pred_len, F_dim)
        
        # 最终预测结果 = 物理粗粒度骨架 + 非线性细粒度微调残差
        y_pred = y_skeleton + y_residual
        
        return y_pred, y_temporal, y_spatial, z1, z2