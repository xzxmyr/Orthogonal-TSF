# ==========================================
# exp/exp_main.py
# ==========================================
import torch
import torch.nn as nn
from data_provider.data_loader import data_provider
from models import EntangledBaselineModel, DisentangledProposedModel
from losses import OrthogonalLoss
from utils import Evaluator  # 只导入纯文本指标评估器

class Exp_Main:
    """
    实验主执行器 (Experiment Execution Engine)
    
    职责：
    1. 根据配置自动切换和编译不同的模型架构（Baseline 或 Proposed）。
    2. 控制模型的训练前向传播、梯度反向传播与优化器更新。
    3. 控制验证集/测试集的数据输入流，并在推理结束时将数据打包交给 Evaluator。
    """
    def __init__(self, config):
        self.config = config
        # 自动化分配算力设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 建立模型注册表，极大方便后续横向扩展前沿 Backbone（如 PatchTST）
        self.model_dict = {
            'baseline': EntangledBaselineModel,
            'proposed': DisentangledProposedModel
        }
        
    def _build_model(self, model_name):
        """内部私有函数：负责动态实例化模型并搬运至指定算力设备"""
        model = self.model_dict[model_name](
            seq_len=self.config.seq_len,
            num_features=self.config.num_features,
            pred_len=self.config.pred_len,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        return model

    def train(self, model_name):
        """
        模型核心训练引擎
        """
        model = self._build_model(model_name)
        train_loader = data_provider(self.config, flag='train')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion_mse = nn.MSELoss()
        criterion_ort = OrthogonalLoss()
        
        print(f"\n>>> 正在训练流水线: {model_name.upper()} 架构...")
        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                # 分流控制前向传播与损失计算逻辑
                if model_name == 'baseline':
                    preds = model(x)
                    loss = criterion_mse(preds, y)
                elif model_name == 'proposed':
                    preds, z1, z2, _ = model(x)
                    loss_mse = criterion_mse(preds, y)
                    loss_ort = criterion_ort(z1, z2)
                    # 引入几何正交约束惩罚项
                    loss = loss_mse + self.config.lambda_ort * loss_ort
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            
            print(f"   Epoch [{epoch+1}/{self.config.epochs}] | 训练集聚合总损失: {total_loss / len(train_loader.dataset):.4f}")
        return model

    def evaluate(self, baseline_model, proposed_model):
        """
        测试集推理与数据审计流管理
        """
        print("\n>>> 正在启动验证集/测试集多维闭环审计...")
        test_loader = data_provider(self.config, flag='test')
        
        # 强制切换为评估模式（锁死 Dropout 和 BatchNormalization）
        baseline_model.eval()
        proposed_model.eval()

        # 1. 实例化全栈学术评测类
        evaluator = Evaluator(self.config)
        evaluator.reset()

        # 2. 禁绝梯度计算，开启高阶推理加速
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 捕获黑盒与解耦模型的预测输出
                preds_base = baseline_model(x)
                preds_prop, z1, z2, gates = proposed_model(x)
                
                # 3. 将所有原始指标及中间层特征流分批塞入评估器进行累加统计
                evaluator.update(
                    y_true=y, 
                    preds_base=preds_base, 
                    preds_prop=preds_prop, 
                    z1=z1, 
                    z2=z2, 
                    gates=gates
                )

        # 4. 触发纯文本文字大报告
        evaluator.report()
        
        # ─── 终极解耦核心：把装满测试集文字数据的实例原样吐还给 main.py，供其进行控流绘图 ───
        return evaluator