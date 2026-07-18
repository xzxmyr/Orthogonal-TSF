# losses/info_losses.py
import torch
import torch.nn as nn

class GatedConditionalCLUB(nn.Module):
    def __init__(self, hidden_dim=64, condition_dim=672):
        super(GatedConditionalCLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(hidden_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(hidden_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim) # 去除死板的 Tanh，允许方差自适应放大以平抑能量
        )

    def get_log_likelihood(self, z1, z2, c):
        inputs = torch.cat([z1, c], dim=-1)
        mu = self.p_mu(inputs)
        logvar = self.p_logvar(inputs)
        
        # 🆕 【安全阀 1】：将原先的 torch.sum 改为 torch.mean，把 64 维的膨胀直接缩减 64 倍
        log_likelihood = -0.5 * (((z2 - mu) ** 2) * torch.exp(-torch.clamp(logvar, -5, 5)) + logvar)
        return torch.mean(torch.mean(log_likelihood, dim=-1))

    def forward(self, z1, z2, c):
        B = z1.size(0)
        inputs = torch.cat([z1, c], dim=-1)
        mu = self.p_mu(inputs)
        logvar = torch.clamp(self.p_logvar(inputs), -5, 5) # 约束方差界限
        
        # 🆕 【安全阀 2】：全部由 sum 变更为 mean，强制将上界拉回 [0, 5] 的理性区间
        pos_log_lr = -0.5 * torch.mean(((z2 - mu) ** 2) * torch.exp(-logvar) + logvar, dim=-1)
        
        mu_expand = mu.unsqueeze(1)      
        logvar_expand = logvar.unsqueeze(1) 
        z2_expand = z2.unsqueeze(0)      
        
        neg_log_lr = -0.5 * torch.mean(((z2_expand - mu_expand) ** 2) * torch.exp(-logvar_expand) + logvar_expand, dim=-1)
        mean_pos = torch.mean(pos_log_lr)
        mean_neg = (torch.sum(neg_log_lr) - torch.sum(pos_log_lr)) / (B * (B - 1))
        
        cmi_upper_bound = mean_pos - mean_neg
        return cmi_upper_bound