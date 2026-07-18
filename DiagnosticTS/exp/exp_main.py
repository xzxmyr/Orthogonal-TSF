# exp/exp_main.py
import torch
import torch.nn as nn
from data_provider.data_loader import data_provider
from models import EntangledBaselineModel, DiagnosticTS
from losses.info_losses import GatedConditionalCLUB # 导入全新 CMI 估计器
from utils import Evaluator

class Exp_Main:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dict = {
            'baseline': EntangledBaselineModel,
            'proposed': DiagnosticTS
        }
        
    def _build_model(self, model_name):
        return self.model_dict[model_name](
            seq_len=self.config.seq_len,
            num_features=self.config.num_features,
            pred_len=self.config.pred_len,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)

    def train(self, model_name):
        model = self._build_model(model_name)
        train_loader = data_provider(self.config, flag='train')
        criterion_mse = nn.MSELoss()
        
        print(f"\n>>> 正在训练流水线: {model_name.upper()} 架构...")
        
        if model_name == 'baseline':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    preds = model(x)
                    loss = criterion_mse(preds, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * x.size(0)
                print(f"   Epoch [{epoch+1}/{self.config.epochs}] | Baseline 训练集总损失: {total_loss / len(train_loader.dataset):.4f}")
                
        elif model_name == 'proposed':
            # 1. 为 Proposed 主网络和 MINE 估计网络分别建立独立的优化器
            condition_dim = self.config.seq_len * self.config.num_features
            cmi_estimator = GatedConditionalCLUB(
                hidden_dim=self.config.hidden_dim, 
                condition_dim=condition_dim
            ).to(self.device)
    
            optimizer_model = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            # 判别器采用独立轻量学习率
            optimizer_club = torch.optim.Adam(cmi_estimator.parameters(), lr=0.0005)
            criterion_mse = nn.MSELoss()
    
            for epoch in range(self.config.epochs):
                model.train()
                cmi_estimator.train()
                total_loss, total_mse, total_cmi = 0, 0, 0
                
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    B = x.size(0)
                    # 将原始输入平铺作为条件 C
                    c = x.reshape(B, -1)
                    
                    # ─── 步骤一: 前向传播提取特征流 ───
                    preds, _, _, z1, z2 = model(x)
                    
                    # ─── 步骤二: 更新 CLUB 估计器 (最大化对数似然以精准锁定边界) ───
                    optimizer_club.zero_grad()
                    # 用 detach 截断传给主模型的梯度，专心更新估计器参数
                    log_likelihood = cmi_estimator.get_log_likelihood(z1.detach(), z2.detach(), c)
                    club_loss = -log_likelihood # 负号变最大化
                    club_loss.backward()
                    torch.nn.utils.clip_grad_norm_(cmi_estimator.parameters(), max_norm=1.0)
                    optimizer_club.step()
                    
                    # ─── 步骤三: 更新 Proposed 主模型 (最小化预测误差 + 最小化 CMI 上界) ───
                    optimizer_model.zero_grad()
                    # 重新前向计算以保持计算图连接
                    preds, _, _, z1, z2 = model(x)
                    loss_mse = criterion_mse(preds, y)
                    
                    # 计算当前的条件互信息上界惩罚
                    current_cmi = cmi_estimator(z1, z2, c)
                    # 联合优化平衡杠杆 (借用 config.lambda_ort 作为系数)
                    loss_model = loss_mse + self.config.lambda_cmi * current_cmi
                    
                    loss_model.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer_model.step()
                    
                    total_loss += loss_model.item() * B
                    total_mse += loss_mse.item() * B
                    total_cmi += current_cmi.item() * B
                    
                print(f"   Epoch [{epoch+1}/{self.config.epochs}] | "
                    f"Model_Loss: {total_loss/len(train_loader.dataset):.4f} | "
                    f"MSE: {total_mse/len(train_loader.dataset):.4f} | "
                    f"CLUB_CMI: {total_cmi/len(train_loader.dataset):.4f}")
                    
            # 将估计器动态挂载到模型实体上，方便后续测试集做数据审计
            model.cmi_estimator = cmi_estimator
            
        return model

    def evaluate(self, baseline_model, proposed_model):
        """
        测试集推理与全栈物理/信息论双重因果审计
        """
        print("\n>>> 正在启动验证集/测试集多维闭环条件互信息审计...")
        test_loader = data_provider(self.config, flag='test')
        
        # 强制切换为评估模式
        baseline_model.eval()
        proposed_model.eval()
        if hasattr(proposed_model, 'cmi_estimator'):
            proposed_model.cmi_estimator.eval()

        evaluator = Evaluator(self.config)
        evaluator.reset()

        # 导入刚刚打好物理补丁的全量纲因果审计器
        # 如果你把 audit_entropy_share 放到了其他文件，请确保在这里能正确引用
        from utils.metrics import audit_entropy_share 

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                B = x.size(0)
                
                # 🆕 1. 严格对齐条件 C：在评估时同样构造平铺后的 2维环境背景 [B, 672]
                c = x.reshape(B, -1)
                
                # 捕获黑盒基准预测
                preds_base = baseline_model(x)
                
                # 🆕 2. 接收升级后的 DiagnosticTS 吐出的 5 个核心组件
                # 此时 z1, z2 已经是模型出厂自带 Softplus 纯化好的 2维正定隐表征 [B, hidden_dim]
                preds_prop, y_temporal, y_spatial, z1, z2 = proposed_model(x)
                
                # 🆕 3. 稳健地结算当前的 CLUB 条件互信息上界值
                if hasattr(proposed_model, 'cmi_estimator'):
                    # 严格传入三个对齐的 2维张量：z1(2维), z2(2维), c(2维)
                    cmi_val = proposed_model.cmi_estimator(z1, z2, c).item()
                else:
                    cmi_val = 0.0
                
                # 🆕 4. 实时调用全量纲因果审计器，抽取当前 Batch 每一时间步的时空控制力
                w1_share, w2_share = audit_entropy_share(y_temporal, y_spatial)
                
                # 🆕 5. 将处理好的指标与时空因果份额批量塞入 evaluator 进行全局结算
                evaluator.update_cmi(
                    y_true=y, 
                    preds_base=preds_base, 
                    preds_prop=preds_prop, 
                    cmi_val=cmi_val, 
                    gates=torch.tensor(w1_share).to(self.device) # 用计算好的时间因果份额替代老 gate 缓存
                )

        evaluator.report()
        
        # ─── 将满载胜果的评估实例扔还给主入口 ───
        return evaluator