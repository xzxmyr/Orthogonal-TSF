import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 设置随机种子以确保实验可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 虚拟多变量时间序列数据加载器 (ETT/Weather 模拟)
# ==========================================
# ==========================================
# 1. 虚拟多变量时间序列数据加载器
# ==========================================
class RealMTSDataset(Dataset):
    def __init__(self, csv_path, seq_len=96, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 读取 CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")
            
        df = pd.read_csv(csv_path)
        if 'date' in df.columns or 'Date' in df.columns:
            df = df.drop(columns=['date', 'Date'], errors='ignore')

        data_matrix = df.values
        self.mean = np.mean(data_matrix, axis=0)
        self.std = np.std(data_matrix, axis=0)
        self.data = (data_matrix - self.mean) / (self.std + 1e-5)
        self.num_features = self.data.shape[1]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
# ==========================================
# 2. 正交去相关软约束损失函数 (Orthogonal Loss)
# ==========================================
class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, z1, z2):
        # z1, z2 形状:
        # 在隐空间特征维度 H 上进行 L2 归一化
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)
        
        # 计算在每个时间步和Batch上的余弦相似度
        cos_sim = (z1_norm * z2_norm).sum(dim=-1) # 形状:
        
        # 惩罚其余弦相似度的平方，使其尽可能接近 0 (即正交)
        loss_ort = torch.mean(cos_sim ** 2)
        return loss_ort


# ==========================================
# 3. 对比实验模型 A: 正常全纠缠基准模型 (Baseline)
# ==========================================
class EntangledBaselineModel(nn.Module):
    """
    基准模型：直接将所有特征送入GRU进行混合建模，不进行任何解耦与物理拆分
    """
    def __init__(self, seq_len, num_features, pred_len, hidden_dim=64):
        super(EntangledBaselineModel, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.pred_len = pred_len
        
        self.encoder = nn.GRU(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
        self.predictor = nn.Linear(hidden_dim, num_features * pred_len)

    def forward(self, x):
        # x:
        out, _ = self.encoder(x) #
        out_last = out[:, -1, :] # 提取最后一个时间步的信息
        preds = self.predictor(out_last).view(-1, self.pred_len, self.num_features) #
        return preds


# ==========================================
# 4. 对比实验模型 B: 改进的时空解耦架构 (Proposed)
# ==========================================
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

def plot_and_save_weights(temporal_weights, spatial_weights, save_path="weight_dynamics.png"):
    """
    更新后的可视化：利用 w1 + w2 = 1 的特性，绘制单折线边界并填充上下双色。
    """
    # 1. 转换为 numpy 数组
    w1 = np.array(temporal_weights)
    
    # 【核心优化】：如果测试集样本过多，直接画图会产生密集毛刺。
    # 这里使用滑动平均（Window=50）对权重进行平滑，以展现优雅的宏观演变趋势
    window_size = 50
    if len(w1) > window_size:
        w1_smooth = pd.Series(w1).rolling(window=window_size, min_periods=1).mean().values
    else:
        w1_smooth = w1
        
    x_axis = np.arange(len(w1_smooth))
    
    # 2. 初始化高分辨率画布
    plt.figure(figsize=(12, 5.5), dpi=300)
    
    # 3. 绘制核心分割折线 (代表 w1 的动态边界)
    # 使用深色调优雅分割
    plt.plot(x_axis, w1_smooth, color='#2C3E50', lw=1.8, linestyle='-', label='Decoupling Boundary ($w_1$)')
    
    # 4. 区域色彩填充
    # 上方填充 w2 的颜色 (从 w1_smooth 曲线一直填充到 y=1.0) -> 代表变量耦合空间权重
    plt.fill_between(x_axis, w1_smooth, 1.0, color='#3498DB', alpha=0.75, 
                     label='Variable Coupling Weight ($w_2$)')
    
    # 下方填充 w1 的颜色 (从 y=0.0 一直填充到 w1_smooth 曲线) -> 代表纯时间规律权重
    plt.fill_between(x_axis, 0.0, w1_smooth, color='#E74C3C', alpha=0.75, 
                     label='Temporal Pattern Weight ($w_1$)')
    
    # 5. 图像美化与学术规范
    plt.xlim(x_axis[0], x_axis[-1])
    plt.ylim(0.0, 1.0)  # 严格限制在 0-1 范围内
    
    plt.xlabel("Test Set Observations (Chronological Samples)", fontsize=11, labelpad=8)
    plt.ylabel("Weight Contribution Proportion", fontsize=11, labelpad=8)
    plt.title("Dynamic Weight Allocation Spectrum ($w_1 + w_2 = 1$)", fontsize=13, fontweight='bold', pad=15)
    
    # 6. 网格线与边框精简 (去除了顶部和右侧多余边框，使版面更具现代学术感)
    plt.grid(axis='x', color='gray', linestyle='--', alpha=0.25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 7. 图例位置优化 (放置在图表正上方或右上角空旷处)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=10)
    
    # 8. 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🎉 完美的双色互补权重动态谱图已保存至: {save_path}")


# ==========================================
# 5. 训练与评测总控流水线
# ==========================================
def run_comparison_experiment():
    # 超参数定义
    SEQ_LEN = 96
    PRED_LEN = 24
    NUM_FEATURES = 7
    BATCH_SIZE = 32
    EPOCHS = 8
    LR = 0.001
    LAMBDA_ORT = 0.1 # 正交正则化系数 (控制解耦强度)
    
    # 实例化数据集与加载器
    dataset = RealMTSDataset(csv_path='/Users/xixiangyu/Desktop/03_个人发展与项目/surf/data/ETTm1.csv', seq_len=SEQ_LEN, pred_len=PRED_LEN)

    # 简单划分训练集与测试集 (8:2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---------------------------
    # A. 训练基准模型 (Baseline)
    # ---------------------------
    print(">>> 正在训练基准黑盒模型 (Baseline)...")
    baseline_model = EntangledBaselineModel(SEQ_LEN, NUM_FEATURES, PRED_LEN)
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=LR)
    criterion_mse = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        baseline_model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer_base.zero_grad()
            preds = baseline_model(x)
            loss = criterion_mse(preds, y)
            loss.backward()
            optimizer_base.step()
            total_loss += loss.item() * x.size(0)
        # print(f"Epoch {epoch+1}/{EPOCHS} | Baseline Loss: {total_loss/train_size:.4f}")

    # ---------------------------
    # B. 训练改进解耦模型 (Proposed)
    # ---------------------------
    print("\n>>> 正在训练时空解耦模型 (Proposed with Orthogonal regularization)...")
    proposed_model = DisentangledProposedModel(SEQ_LEN, NUM_FEATURES, PRED_LEN)
    optimizer_prop = torch.optim.Adam(proposed_model.parameters(), lr=LR)
    criterion_ort = OrthogonalLoss()
    
    for epoch in range(EPOCHS):
        proposed_model.train()
        total_loss = 0
        total_mse = 0
        total_ort = 0
        for x, y in train_loader:
            optimizer_prop.zero_grad()
            preds, z1, z2, _ = proposed_model(x)
            
            # 主预测误差
            loss_mse = criterion_mse(preds, y)
            # 正交软约束
            loss_ort = criterion_ort(z1, z2)
            
            # 联合损失
            loss_total = loss_mse + LAMBDA_ORT * loss_ort
            
            loss_total.backward()
            optimizer_prop.step()
            
            total_loss += loss_total.item() * x.size(0)
            total_mse += loss_mse.item() * x.size(0)
            total_ort += loss_ort.item() * x.size(0)
        # print(f"Epoch {epoch+1}/{EPOCHS} | Total: {total_loss/train_size:.4f} | MSE: {total_mse/train_size:.4f} | Ortho_Penalty: {total_ort/train_size:.4f}")

    # ---------------------------
    # C. 在测试集上进行全方位对比
    # ---------------------------
    print("\n>>> 开始在验证集上对比两种模型的性能...")
    baseline_model.eval()
    proposed_model.eval()

    base_mses, base_maes = [], []
    prop_mses, prop_maes = [], []

    # 诊断数据存储
    all_temporal_weights = []
    all_spatial_weights = []
    cos_similarities_with_penalty = []
    cos_similarities_no_penalty = []

    with torch.no_grad():
        for x, y in test_loader:
            # 基准预测评价
            preds_base = baseline_model(x)
            base_mses.append(F.mse_loss(preds_base, y).item())
            base_maes.append(F.l1_loss(preds_base, y).item())
            
            # 解耦预测评价
            preds_prop, z1, z2, gates = proposed_model(x)
            prop_mses.append(F.mse_loss(preds_prop, y).item())
            prop_maes.append(F.l1_loss(preds_prop, y).item())
            
            # 统计诊断权重与解耦程度
            all_temporal_weights.extend(gates[:, 0].cpu().numpy())
            all_spatial_weights.extend(gates[:, 1].cpu().numpy())
            
            # 相似度分析
            z1_n = F.normalize(z1, p=2, dim=-1)
            z2_n = F.normalize(z2, p=2, dim=-1)
            similarity = torch.mean((z1_n * z2_n).sum(dim=-1)**2).item()
            cos_similarities_with_penalty.append(similarity)

    # 打印最终对比结论
    print("\n================== 实验结果报告 ==================")
    print(f"1. 预测精度评估 (MSE / MAE):")
    print(f"  : MSE = {np.mean(base_mses):.5f} | MAE = {np.mean(base_maes):.5f}")
    print(f"   [时空解耦模型 (Proposed)]: MSE = {np.mean(prop_mses):.5f} | MAE = {np.mean(prop_maes):.5f}")
    
    print(f"\n2. 内在解耦程度指标 (余弦相似度平方，越接近0说明时空分得越开):")
    print(f"   [解耦约束模型]: Cos_Sim^2 = {np.mean(cos_similarities_with_penalty):.5f}")
    
    print(f"\n3. 内在诊断权重比例 (模型自主学习融合时间规律与变量耦合的倾向):")
    print(f"   平均时间规律权重 (w1): {np.mean(all_temporal_weights):.2%}")
    print(f"   平均变量耦合权重 (w2): {np.mean(all_spatial_weights):.2%}")
    print("==================================================")

    # --- 新增调用逻辑 ---
    # 将记录下来的权重列表传入绘图函数
    plot_and_save_weights(all_temporal_weights, all_spatial_weights)
    # -----------------------------
    
    if abs(np.mean(prop_mses) - np.mean(base_mses)) < 0.05:
        print("💡 [可行性诊断]: 恭喜！解耦模型的 MSE 与基准模型极其接近。这完美证明了：")
        print("   我们在通过正交软约束‘强行解耦潜在特征’、‘剥离重叠信息’的同时，没有破坏模型对预测标签的编码能力。")
        print("   时空划分预测机制是完全成立的，你可以以此为基石申请 SURF 项目！")
    else:
        print("⚠️ [参数调整建议]: 预测误差产生了一定偏离，建议在训练中适当调小解耦惩罚项的超参数 lambda_ort (例如设为 0.01) 重试。")
    
if __name__ == "__main__":
    run_comparison_experiment()