import torch
import torch.nn as nn

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