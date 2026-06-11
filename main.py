# ==========================================
# main.py (主入口启动脚本)
# ==========================================
import torch
import numpy as np
from config import ExperimentConfig
from exp import Exp_Main
from utils import plot_and_save_weights

def main():
    print("==================================================")
    print("🚀 [Diagnostic AI] 时空解耦时序预测与残差归因系统启动")
    print("==================================================")

    # 1. 初始化所有集中的超参数与路径配置
    config = ExperimentConfig()
    
    # 2. 锁定全局随机种子，确保学术研究与对比实验的严格可复现性
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        # 确保底层卷积或矩阵乘法算法在固定种子下表现一致
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # 3. 实例化全栈实验主执行器 (控制训练、数据分流的总控引擎)
    exp = Exp_Main(config)
    
    # 4. 依次链接并运行两条独立模型的训练流水线
    baseline_model = exp.train('baseline')
    proposed_model = exp.train('proposed')
    
    # 5. 触发跨模型评测审计，输出纯文字结果报告
    # 此时 exp.evaluate() 内部调用 evaluator.report()，进行纯文本统计输出
    evaluator = exp.evaluate(baseline_model, proposed_model)
    
    # 6. ─── 核心控流：由主入口提取统计特征，并触发独立的解耦谱图可视化 ───
    # 检查评估器中是否成功拦截并缓存了测试集的动态门控权重
    if hasattr(evaluator, 'all_temporal_weights') and len(evaluator.all_temporal_weights) > 0:
        print("\n>>> 正在为主入口提取内在诊断特征，开始渲染高分辨率学术谱图...")
        
        # 显式调用已完全解耦的可视化引擎
        plot_and_save_weights(
            temporal_weights=evaluator.all_temporal_weights,
            spatial_weights=evaluator.all_spatial_weights,
            save_path=config.save_path
        )
    else:
        print("\n⚠️ [可视化提示]: 未捕获到有效的门控权重历史，跳过谱图绘制。")

    print("\n🎉 [实验结束]: 所有模块运行完毕，全栈学术审计闭环已达成。")
    print("==================================================")

if __name__ == "__main__":
    main()